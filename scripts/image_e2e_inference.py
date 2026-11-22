"""
Image -> mesh: TRELLIS sparse-structure stage -> V-Flow -> T-Flow.

Outputs into --out_dir:
    <image_id>_cond.png         the conditioning image (foreground crop on black) fed to DINO-v2
    <image_id>_ss.ply           generated 64^3 sparse structure (voxel centers) in [-0.5, 0.5]
    <image_id>_pred.ply         generated vertices (+offset head) in [-0.5, 0.5]
    <image_id>_pred_coords.ply  generated vertex voxel coords in [0, 1024)
    <image_id>_pred.obj         generated mesh (offset vertices, faces) in [-0.5, 0.5]
    <image_id>_pred_coords.obj  generated mesh on integer voxel coords in [0, 1024)

Every image is matted with BiRefNet to find its foreground; an alpha channel
already in the file is ignored (flattened onto white).

Usage:
    python scripts/image_e2e_inference.py --image_dir <dir> --out_dir outputs/image_e2e_run/<dir> \
        [--vert_num 2000] [--ss_steps 25] [--ss_cfg_strength 5.0] \
        [--cfg_strength 3.0] [--vflow_steps 24] [--tflow_steps 50] \
        [--no-fill_open_rings] [--max_ring_size 12]
"""

import argparse
import os
import sys
import time
from collections import Counter

import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import trimesh
from PIL import Image

from models import (
    BackgroundRemover,
    DinoV2Encoder,
    OffsetHead,
    TopoFlowEulerSampler,
    TopologySiTFlow,
    TopologyVAE,
    VertexSLatFlowModel,
    VertFlowEulerCfgSampler,
    VertexVAE,
    VoxelFieldConditioner,
)
from models.trellis_ss import (
    SS_COND_MODEL,
    SS_COND_RESOLUTION,
    SS_DECODER_NAME,
    SS_FLOW_NAME,
    SSFlowEulerSampler,
    load_trellis_model,
)
from modules.sparse import SparseTensor
import utils.logging as logging
from utils.device import add_device_arg, autocast, resolve_device
from utils.export import export_mesh, export_vertex
from utils.image import IMAGE_EXTENSIONS, crop_foreground, load_rgba
from utils.inference import (
    build_voxel_fields,
    decode_vertices,
    edges_to_faces,
    pad_verts,
)
from utils.load import load_latov2_model


def parse_args():
    p = argparse.ArgumentParser(
        description="image -> sparse structure -> vertex + topology generation"
    )
    p.add_argument("--image_dir", required=True, help="directory of input images")
    p.add_argument(
        "--out_dir", required=True, help="output directory for the PLYs / OBJs"
    )
    p.add_argument("--vflow_ckpt", default=os.path.join(ROOT, "ckpt", "vflow.pt"))
    p.add_argument("--vvae_ckpt", default=os.path.join(ROOT, "ckpt", "vvae.pt"))
    p.add_argument(
        "--offset_head_ckpt", default=os.path.join(ROOT, "ckpt", "offset_head.pt")
    )
    p.add_argument("--tflow_ckpt", default=os.path.join(ROOT, "ckpt", "tflow.pt"))
    p.add_argument("--tvae_ckpt", default=os.path.join(ROOT, "ckpt", "tvae.pt"))
    p.add_argument(
        "--voxel_encoder_ckpt",
        default=os.path.join(ROOT, "ckpt", "voxel_encoder.pt"),
    )
    p.add_argument(
        "--trellis_dir",
        default=os.path.join(ROOT, "ckpt", "trellis"),
        help=(
            "TRELLIS sparse-structure flow / decoder checkpoints; reused when "
            "present, downloaded otherwise"
        ),
    )
    p.add_argument(
        "--rembg_dir",
        default=os.path.join(ROOT, "ckpt", "rembg"),
        help=(
            "BiRefNet background-removal model applied to every image; reused "
            "when present, downloaded otherwise (its bundled code is executed)"
        ),
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--num_samples", type=int, default=None, help="only run the first N images"
    )
    p.add_argument("--inference_threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    add_device_arg(p)
    # conditioning image
    p.add_argument(
        "--crop_padding",
        type=float,
        default=1.2,
        help="crop side relative to the foreground bounding box (training value: 1.2)",
    )
    # sparse-structure flow sampling (TRELLIS defaults)
    p.add_argument("--ss_steps", type=int, default=25, help="SS-Flow Euler steps")
    p.add_argument("--ss_cfg_strength", type=float, default=5.0)
    p.add_argument(
        "--ss_cfg_interval",
        type=float,
        nargs=2,
        default=(0.5, 1.0),
        metavar=("T_MIN", "T_MAX"),
        help="apply SS-Flow guidance only while t is inside this interval",
    )
    p.add_argument("--ss_rescale_t", type=float, default=3.0)
    # vertex flow sampling
    p.add_argument("--vflow_steps", type=int, default=24, help="V-Flow Euler steps")
    p.add_argument("--cfg_strength", type=float, default=3.0)
    p.add_argument("--rescale_t", type=float, default=1.0)
    p.add_argument("--vert_num", type=int, default=2000, help="target vertex count")
    # topology flow sampling / decoding
    p.add_argument("--tflow_steps", type=int, default=50, help="T-Flow Euler steps")
    p.add_argument("--edge_threshold", type=float, default=0.0)
    p.add_argument("--chunk_size", type=int, default=20000)
    p.add_argument(
        "--fill_open_rings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "post-process: triangulate chordless rings of open edges (edges with "
            "fewer than two faces) by minimum area, not the voxel support filter"
        ),
    )
    p.add_argument(
        "--max_ring_size",
        type=int,
        default=12,
        help="largest open-edge ring (in edges) that --fill_open_rings closes",
    )
    p.add_argument(
        "--dino_hub_dir",
        default=os.path.join(ROOT, "ckpt", "dinov2"),
        help="torch.hub cache for DINO-v2; reused when present, downloaded otherwise",
    )
    args = p.parse_args()
    if args.num_samples is not None and args.num_samples <= 0:
        args.num_samples = None  # <= 0 means "all", not python slice semantics
    return args


def ss_to_lato_voxels(occupancy: torch.Tensor) -> torch.Tensor:
    """(B, 1, R, R, R) TRELLIS occupancy -> sorted (N, 4) int [batch, x, y, z].

    TRELLIS grids are Z-up while LATO.2 meshes are Y-up (glTF), so voxels are
    rotated by (x, y, z) -> (x, z, -y).
    """
    res = occupancy.shape[-1]
    c = torch.argwhere(occupancy[:, 0])  # (N, 4): batch, x, y, z
    c = torch.stack([c[:, 0], c[:, 1], c[:, 3], res - 1 - c[:, 2]], dim=1)
    order = torch.argsort(((c[:, 0] * res + c[:, 1]) * res + c[:, 2]) * res + c[:, 3])
    return c[order].int()


def main():
    logging.info("Image end-to-end inference starting...")

    args = parse_args()
    device = resolve_device(args.device)
    logging.info(f"device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # stage 0: sparse structure generation (TRELLIS)
    ss_flow = load_trellis_model(SS_FLOW_NAME, args.trellis_dir, device)
    ss_decoder = load_trellis_model(SS_DECODER_NAME, args.trellis_dir, device)
    # stage 1: vertex generation
    vflow, vflow_cfg = load_latov2_model(VertexSLatFlowModel, args.vflow_ckpt, device)
    vvae, vvae_cfg = load_latov2_model(VertexVAE, args.vvae_ckpt, device)
    offset_head, _ = load_latov2_model(OffsetHead, args.offset_head_ckpt, device)
    # stage 2: topology generation
    tflow, tflow_cfg = load_latov2_model(TopologySiTFlow, args.tflow_ckpt, device)
    tvae, _ = load_latov2_model(TopologyVAE, args.tvae_ckpt, device)
    voxel_encoder, venc_cfg = load_latov2_model(
        VoxelFieldConditioner, args.voxel_encoder_ckpt, device
    )

    res = vvae_cfg["resolution"]
    min_res = vvae_cfg["min_resolution"]
    latent_dim = vflow_cfg["latent_dim"]
    density_max = vflow_cfg["max_vertex_num"]
    img_res = vflow_cfg["image_resolution"]
    z_dim = int(tflow_cfg["args"]["z_dim"])
    num_discrete = int(tflow_cfg["args"]["num_discrete"])
    max_vertices = int(tflow_cfg["args"]["max_vertices"])
    latent_scale = float(tflow_cfg["latent_scale"])
    voxel_res = int(venc_cfg["resolution"])
    if num_discrete != res:
        raise ValueError(
            f"T-Flow num_discrete={num_discrete} != V-VAE resolution={res}; "
            "the generated vertex voxels would be in the wrong coordinate space."
        )
    if voxel_res != min_res:
        raise ValueError(
            f"voxel encoder resolution={voxel_res} != V-VAE min_resolution={min_res}; "
            "both stages must share the same active-voxel conditioning grid."
        )
    if (vflow_cfg["dino_version"], img_res) != (SS_COND_MODEL, SS_COND_RESOLUTION):
        raise ValueError(
            f"V-Flow is conditioned on {vflow_cfg['dino_version']}@{img_res} but the "
            f"TRELLIS SS-Flow needs {SS_COND_MODEL}@{SS_COND_RESOLUTION}; the two "
            "stages can no longer share one conditioning image."
        )

    dino = (
        DinoV2Encoder(
            model_name=vflow_cfg["dino_version"],
            hub_dir=args.dino_hub_dir,
            img_res=img_res,
        )
        .to(device)
        .eval()
    )
    logging.info(f"loaded {vflow_cfg['dino_version']} from {args.dino_hub_dir}")
    remove_background = BackgroundRemover(args.rembg_dir, device)
    logging.info(f"loaded background remover from {remove_background.model_dir}")
    ss_sampler = SSFlowEulerSampler()
    vertex_sampler = VertFlowEulerCfgSampler()
    topo_sampler = TopoFlowEulerSampler()

    files = sorted(
        f
        for f in os.listdir(args.image_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )
    if args.num_samples is not None:
        files = files[: args.num_samples]
    if not files:
        raise ValueError(f"no image files ({IMAGE_EXTENSIONS}) under {args.image_dir}")
    dupes = sorted(
        s for s, c in Counter(os.path.splitext(f)[0] for f in files).items() if c > 1
    )
    if dupes:
        logging.warning(
            f"WARNING: {len(dupes)} duplicate image basename(s) — later samples will overwrite earlier outputs."
        )
    logging.info(
        f"{len(files)} images from {args.image_dir} "
        f"(ss_steps={args.ss_steps}, ss_cfg={args.ss_cfg_strength}, "
        f"vflow_steps={args.vflow_steps}, cfg={args.cfg_strength}, "
        f"vert_num={args.vert_num}, density_max={density_max}, "
        f"tflow_steps={args.tflow_steps}, seed={args.seed}) -> {args.out_dir}"
    )

    n_ok = n_no_topo = n_fail = 0
    t_start = time.time()
    qbar = tqdm.tqdm(
        range(0, len(files), args.batch_size),
        desc="inference",
        unit="batch",
        dynamic_ncols=True,
    )
    for start in qbar:
        names, images = [], []
        for f in files[start : start + args.batch_size]:
            name = os.path.splitext(f)[0]
            try:
                rgba = load_rgba(os.path.join(args.image_dir, f), remove_background)
                images.append(crop_foreground(rgba, args.crop_padding, img_res))
                names.append(name)
            except Exception as e:
                n_fail += 1
                logging.error(f"{name}: FAILED during preprocessing: {e}")
        if not names:
            continue

        # ---- stage 0: SS-Flow on the image -> SS decoder -> 64^3 active voxels ----
        with torch.no_grad():
            cond = dino(np.stack(images)).float()
            neg_cond = torch.zeros_like(cond)
            ss_res = ss_flow.resolution
            z_ss = ss_sampler.sample(
                model=ss_flow,
                noise=torch.randn(
                    len(names), ss_flow.in_channels, ss_res, ss_res, ss_res, device=device
                ),
                cond=cond,
                neg_cond=neg_cond,
                steps=args.ss_steps,
                cfg_strength=args.ss_cfg_strength,
                cfg_interval=args.ss_cfg_interval,
                rescale_t=args.ss_rescale_t,
            )
            occupancy = ss_decoder(z_ss) > 0  # (B, 1, R, R, R)
        if occupancy.shape[-1] != min_res:
            raise ValueError(
                f"SS decoder resolution={occupancy.shape[-1]} != V-VAE "
                f"min_resolution={min_res}; V-Flow expects {min_res}^3 active voxels."
            )
        coords = ss_to_lato_voxels(occupancy).cpu()
        voxels = [coords[coords[:, 0] == b, 1:] for b in range(len(names))]
        keep = [b for b, v in enumerate(voxels) if v.shape[0] > 0]
        for b in sorted(set(range(len(names))) - set(keep)):
            n_fail += 1
            logging.error(f"{names[b]}: empty sparse structure; skipping.")
        if not keep:
            continue
        names = [names[b] for b in keep]
        images = [images[b] for b in keep]
        voxels = [voxels[b] for b in keep]
        cond, neg_cond = cond[keep], neg_cond[keep]
        min_active = torch.cat(
            [
                torch.cat(
                    [torch.full((v.shape[0], 1), i, dtype=torch.int32), v], dim=1
                )
                for i, v in enumerate(voxels)
            ],
            dim=0,
        )
        for name, image, vox in zip(names, images, voxels):
            Image.fromarray(image).save(os.path.join(args.out_dir, f"{name}_cond.png"))
            trimesh.PointCloud(
                (vox.numpy().astype(np.float64) + 0.5) / min_res - 0.5
            ).export(os.path.join(args.out_dir, f"{name}_ss.ply"))

        # ---- stage 1: V-Flow on the 64^3 active voxels -> V-VAE vertex decode ----
        density = torch.full(
            (len(names),),
            float(args.vert_num) / density_max * 1000.0,
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad(), autocast(device):
            min_active_coords = min_active.to(device)
            noise = SparseTensor(
                coords=min_active_coords.int(),
                feats=torch.randn(
                    min_active_coords.shape[0], latent_dim, device=device
                ),
            )
            z_pred = vertex_sampler.sample(
                model=vflow,
                noise=noise,
                cond=cond,
                neg_cond=neg_cond,
                steps=args.vflow_steps,
                cfg_strength=args.cfg_strength,
                rescale_t=args.rescale_t,
                density=density,
            )
            pred_coords, pred_offsets = decode_vertices(
                vvae, offset_head, z_pred, args.inference_threshold
            )

        keep_idx, verts_list, offsets_list = [], [], []
        for b, name in enumerate(names):
            pred_sel = pred_coords[:, 0] == b
            vert_int = pred_coords[pred_sel, 1:].long()
            vert_off = pred_offsets[pred_sel]
            export_vertex(
                args.out_dir,
                name,
                type_name="pred",
                vert_int=vert_int.numpy(),
                vert_offsets=vert_off.numpy(),
                resolution=res,
            )
            num_pred = int(vert_int.shape[0])
            if num_pred < 3:
                n_no_topo += 1
                logging.warning(
                    f"{name}: only {num_pred} generated vertices; skipping topology."
                )
            elif num_pred > max_vertices:
                n_no_topo += 1
                logging.warning(
                    f"{name}: {num_pred} generated vertices exceed T-Flow "
                    f"max_vertices={max_vertices}; skipping topology."
                )
            else:
                keep_idx.append(b)
                verts_list.append(vert_int)
                offsets_list.append(vert_off)
        if not keep_idx:
            continue

        # ---- stage 2: T-Flow on the generated vertices -> T-VAE edge decode ----
        with torch.no_grad():
            verts, mask, lengths = pad_verts(verts_list, device)
            voxel_list = [voxels[b].long() for b in keep_idx]
            field = build_voxel_fields(voxel_list, voxel_res, device)  # (B', R, R, R)
            cond_vox = voxel_encoder(field)  # (B', R'^3, cond_in_dim)

            z0 = torch.randn(verts.shape[0], verts.shape[1], z_dim, device=device)
            z_flow = topo_sampler.sample(
                model=tflow,
                noise=z0,
                verts=verts,
                mask=mask,
                cond=cond_vox,
                steps=args.tflow_steps,
            )
            z = z_flow.float() / latent_scale

            with autocast(device):
                edges_list = tvae.decode(
                    z,
                    verts=verts,
                    verts_mask=mask,
                    chunk_size=args.chunk_size,
                    threshold=args.edge_threshold,
                )

        for k, b in enumerate(keep_idx):
            name = names[b]
            faces = edges_to_faces(
                edges_list[k],
                lengths[k],
                verts_list[k].numpy(),
                fill_open_rings=args.fill_open_rings,
                max_ring_size=args.max_ring_size,
            )
            if faces.shape[0] == 0:
                n_no_topo += 1
                logging.warning(
                    f"{name}: no faces decoded; the _pred PLYs are the only outputs."
                )
                continue
            export_mesh(
                args.out_dir,
                name,
                vert_int=verts_list[k].numpy(),
                vert_offsets=offsets_list[k].numpy(),
                faces=faces,
                resolution=res,
            )
            n_ok += 1

    logging.info(
        f"done: {n_ok} ok, {n_no_topo} without topology, {n_fail} failed "
        f"in {time.time() - t_start:.0f}s -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
