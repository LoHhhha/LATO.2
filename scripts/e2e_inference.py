"""
Outputs into --out_dir:
    <mesh_id>_pred.ply          generated vertices (+offset head) in [-0.5, 0.5]
    <mesh_id>_pred_coords.ply   generated vertex voxel coords in [0, 1024)
    <mesh_id>_pred.obj          generated mesh (offset vertices, faces) in [-0.5, 0.5]
    <mesh_id>_pred_coords.obj   generated mesh on integer voxel coords in [0, 1024)
    <mesh_id>_render.png        the conditioning view fed to DINO-v2

Usage:
    python scripts/e2e_inference.py --mesh_dir <dir> --out_dir outputs/e2e_run/<dir> \
        [--vert_num 2000] [--cfg_strength 3.0] [--vflow_steps 24] [--tflow_steps 50] \
        [--render_azimuth 45 --render_elevation 30] [--no-fill_quad_rings]
"""

import argparse
import os
import sys
import time
from collections import Counter
from functools import partial

import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-root")
os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)
# Open3D headless rendering: without this the default EGL platform can hang
# (e.g. when every GPU is busy or no display device is exposed).
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import torch
import trimesh
from PIL import Image
from torch.utils.data import DataLoader

from dataset.voxel_dataset import VoxelVertexDataset, collate_fn
from models import (
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
from modules.sparse import SparseTensor
import utils.logging as logging
from utils.export import export_vertex
from utils.inference import (
    build_voxel_fields,
    compute_density,
    decode_vertices,
    edges_to_faces,
    pad_verts,
    worker_init,
)
from utils.load import load_latov2_model


def parse_args():
    p = argparse.ArgumentParser(
        description="end-to-end vertex + topology generation inference"
    )
    p.add_argument("--mesh_dir", required=True, help="directory of input meshes")
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
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--num_samples", type=int, default=None, help="only run the first N meshes"
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--inference_threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    # vertex flow sampling
    p.add_argument("--vflow_steps", type=int, default=24, help="V-Flow Euler steps")
    p.add_argument("--cfg_strength", type=float, default=3.0)
    p.add_argument("--rescale_t", type=float, default=1.0)
    # vertex-count density conditioning
    p.add_argument("--vert_num", type=int, default=2000, help="target vertex count")
    p.add_argument(
        "--use_gt_vert_count",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="condition on the GT quantized vertex count instead of --vert_num",
    )
    p.add_argument(
        "--scaler",
        type=float,
        default=1.0,
        help="multiplier on the GT count when --use_gt_vert_count",
    )
    p.add_argument("--min_verts", type=float, default=200.0)
    p.add_argument("--max_verts", type=float, default=5000.0)
    # topology flow sampling / decoding
    p.add_argument("--tflow_steps", type=int, default=50, help="T-Flow Euler steps")
    p.add_argument("--edge_threshold", type=float, default=0.0)
    p.add_argument("--chunk_size", type=int, default=20000)
    p.add_argument(
        "--fill_quad_rings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "post-process: split chordless 4-vertex rings into two triangles "
            "(pure topology, not the voxel support filter)"
        ),
    )
    # conditioning render
    p.add_argument("--render_azimuth", type=float, default=45.0)
    p.add_argument("--render_elevation", type=float, default=30.0)
    p.add_argument("--img_res", type=int, default=518)
    p.add_argument(
        "--dino_hub_dir",
        default=os.path.join(ROOT, "ckpt", "dinov2"),
        help="torch.hub cache for DINO-v2; reused when present, downloaded otherwise",
    )
    args = p.parse_args()
    if args.num_samples is not None and args.num_samples <= 0:
        args.num_samples = None  # <= 0 means "all", not python slice semantics
    return args


def export_mesh(out_dir, base_name, vert_int, vert_offsets, faces, resolution):
    """OBJ pair matching export_vertex's PLY conventions (offset verts / int coords)."""
    res = float(resolution)
    vert_with_offset = (
        vert_int.astype(np.float64) / res
        - 0.5
        + vert_offsets.astype(np.float64) / (res * 2.0)
    )
    trimesh.Trimesh(vertices=vert_with_offset, faces=faces).export(
        os.path.join(out_dir, f"{base_name}_pred.obj")
    )
    trimesh.Trimesh(vertices=vert_int.astype(np.float64), faces=faces).export(
        os.path.join(out_dir, f"{base_name}_pred_coords.obj")
    )


def main():
    logging.info("End-to-end inference starting...")

    args = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

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

    dino = (
        DinoV2Encoder(
            model_name=vflow_cfg["dino_version"],
            hub_dir=args.dino_hub_dir,
            img_res=vflow_cfg["image_resolution"],
        )
        .to(device)
        .eval()
    )
    logging.info(f"loaded {vflow_cfg['dino_version']} from {args.dino_hub_dir}")
    vertex_sampler = VertFlowEulerCfgSampler()
    topo_sampler = TopoFlowEulerSampler()

    dataset = VoxelVertexDataset(
        root_dir=args.mesh_dir,
        resolution=res,
        min_resolution=min_res,
        need_encoder_inputs=False,
        num_samples=args.num_samples,
        render=True,
        img_res=args.img_res,
        render_azimuth=args.render_azimuth,
        render_elevation=args.render_elevation,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, resolution=res, min_resolution=min_res),
        num_workers=args.num_workers,
        pin_memory=True,
        # EGL rendering hangs inside fork-ed children of a CUDA-initialized
        # parent; spawn gives each worker a clean process for its EGL context.
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
        worker_init_fn=worker_init if args.num_workers > 0 else None,
    )
    dupes = sorted(
        s
        for s, c in Counter(os.path.splitext(f)[0] for f in dataset.files).items()
        if c > 1
    )
    if dupes:
        logging.warning(
            f"WARNING: {len(dupes)} duplicate mesh basename(s) — later samples will overwrite earlier outputs."
        )
    logging.info(
        f"{len(dataset)} meshes from {args.mesh_dir} "
        f"(vflow_steps={args.vflow_steps}, cfg={args.cfg_strength}, "
        f"vert_num={args.vert_num}, use_gt_vert_count={args.use_gt_vert_count}, "
        f"scaler={args.scaler}, density_max={density_max}, tflow_steps={args.tflow_steps}, "
        f"view=az{args.render_azimuth}/el{args.render_elevation}, seed={args.seed}) "
        f"-> {args.out_dir}"
    )

    n_ok = n_no_topo = n_fail = 0
    t_start = time.time()
    qbar = tqdm.tqdm(loader, desc="inference", unit="batch", dynamic_ncols=True)
    for batch in qbar:
        for err in batch["errors"]:
            n_fail += 1
            logging.error(
                f"{err['name']}: FAILED during preprocessing: {err['error'].splitlines()[0]}"
            )
        if "name" not in batch:
            continue

        density = compute_density(batch, args, density_max, device)
        with torch.no_grad():
            cond = dino(np.stack(batch["image"])).float()
            neg_cond = torch.zeros_like(cond)

        # ---- stage 1: V-Flow on the 64^3 active voxels -> V-VAE vertex decode ----
        min_active = batch[f"active_voxels_{min_res}"]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
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
        for b, name in enumerate(batch["name"]):
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
            Image.fromarray(batch["image"][b]).save(
                os.path.join(args.out_dir, f"{name}_render.png")
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
            voxel_list = [
                min_active[min_active[:, 0] == b, 1:].long() for b in keep_idx
            ]
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

            with torch.autocast("cuda", dtype=torch.bfloat16):
                edges_list = tvae.decode(
                    z,
                    verts=verts,
                    verts_mask=mask,
                    chunk_size=args.chunk_size,
                    threshold=args.edge_threshold,
                )

        for k, b in enumerate(keep_idx):
            name = batch["name"][b]
            faces = edges_to_faces(edges_list[k], lengths[k], args.fill_quad_rings)
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
