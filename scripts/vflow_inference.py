"""
Outputs PLYs and renders into --out_dir:
    <mesh_id>_gt_coords.ply     GT vertex voxel coords in [0, 1024)
    <mesh_id>_gt.ply            GT vertices (+GT offsets) in [-0.5, 0.5],
    <mesh_id>_recon_coords.ply  reconstructed vertex voxel coords in [0, 1024) (if --reconstruct)
    <mesh_id>_recon.ply         reconstructed vertices (+offset head) in [-0.5, 0.5] (if --reconstruct)
    <mesh_id>_pred_coords.ply   flow-generated vertex voxel coords in [0, 1024)
    <mesh_id>_pred.ply          flow-generated vertices (+offset head) in [-0.5, 0.5]
    <mesh_id>_render.png        the conditioning view fed to DINO-v2

Usage:
    python scripts/vflow_inference.py --mesh_dir <dir> --out_dir outputs/vflow_run/<dir> \
        [--vert_num 2000] [--cfg_strength 3.0] [--steps 24] \
        [--render_azimuth 45 --render_elevation 30] [--reconstruct]
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
from PIL import Image
from torch.utils.data import DataLoader

from dataset.voxel_dataset import VoxelVertexDataset, collate_fn
from models import (
    DinoV2Encoder,
    OffsetHead,
    VDFEncoder,
    VertexVAE,
    VertexSLatFlowModel,
    VertFlowEulerCfgSampler,
)
from modules.sparse import SparseTensor
import utils.logging as logging
from utils.export import export_vertex
from utils.inference import compute_density, decode_vertices, worker_init
from utils.load import load_latov2_model


def parse_args():
    p = argparse.ArgumentParser(description="V-Flow vertex generation inference")
    p.add_argument("--mesh_dir", required=True, help="directory of input meshes")
    p.add_argument(
        "--out_dir", required=True, help="output directory for the PLYs / renders"
    )
    p.add_argument("--vflow_ckpt", default=os.path.join(ROOT, "ckpt", "vflow.pt"))
    p.add_argument("--vvae_ckpt", default=os.path.join(ROOT, "ckpt", "vvae.pt"))
    p.add_argument(
        "--vdf_encoder_ckpt", default=os.path.join(ROOT, "ckpt", "vdf_encoder.pt")
    )
    p.add_argument(
        "--offset_head_ckpt", default=os.path.join(ROOT, "ckpt", "offset_head.pt")
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--num_samples", type=int, default=None, help="only run the first N meshes"
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pc_sample_number", type=int, default=819200)
    p.add_argument("--sample_type", choices=["dora", "uniform"], default="dora")
    p.add_argument("--inference_threshold", type=float, default=0.5)
    p.add_argument(
        "--reconstruct",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also reconstruct the vertex using V-VAE",
    )
    p.add_argument(
        "--sample_posterior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sample the GT posterior (training-style) instead of taking its mode",
    )
    p.add_argument("--seed", type=int, default=42)
    # flow sampling
    p.add_argument("--steps", type=int, default=24, help="Euler steps")
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


def main():
    logging.info("V-Flow inference starting...")
    
    args = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    vflow, vflow_cfg = load_latov2_model(VertexSLatFlowModel, args.vflow_ckpt, device)
    vvae, vvae_cfg = load_latov2_model(VertexVAE, args.vvae_ckpt, device)
    vdf_encoder, _ = load_latov2_model(VDFEncoder, args.vdf_encoder_ckpt, device)
    offset_head, _ = load_latov2_model(OffsetHead, args.offset_head_ckpt, device)
    res = vvae_cfg["resolution"]
    min_res = vvae_cfg["min_resolution"]
    latent_dim = vflow_cfg["latent_dim"]
    density_max = vflow_cfg["max_vertex_num"]

    dino = (
        DinoV2Encoder(
            model_name=vflow_cfg["dino_version"],
            hub_dir=args.dino_hub_dir,
            img_res=vflow_cfg["image_resolution"],
        )
        .to(device)
        .eval()
    )
    logging.info(
        f"loaded {vflow_cfg['dino_version']} from {args.dino_hub_dir}"
    )
    sampler = VertFlowEulerCfgSampler()

    dataset = VoxelVertexDataset(
        root_dir=args.mesh_dir,
        resolution=res,
        min_resolution=min_res,
        pc_sample_number=args.pc_sample_number,
        sample_type=args.sample_type,
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
        f"(steps={args.steps}, cfg={args.cfg_strength}, vert_num={args.vert_num}, "
        f"use_gt_vert_count={args.use_gt_vert_count}, scaler={args.scaler}, density_max={density_max}, "
        f"view=az{args.render_azimuth}/el{args.render_elevation}, seed={args.seed}) "
        f"-> {args.out_dir}"
    )

    n_ok = n_fail = 0
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

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if args.reconstruct:
                point_cloud = batch[f"point_cloud_{res}"].to(device)
                vertex_added_active_coords = batch[f"vertex_added_active_voxels_{res}"].to(device)
                feats = vdf_encoder(
                    p=point_cloud,
                    sparse_coords=vertex_added_active_coords,
                    res=res,
                    bbox_size=(-0.5, 0.5),
                )
                z_gt, _ = vvae.encode(
                    SparseTensor(feats=feats, coords=vertex_added_active_coords.int()),
                    sample_posterior=args.sample_posterior,
                )
            
            min_active_coords = batch[f"active_voxels_{min_res}"].to(device)
            noise = SparseTensor(
                coords=min_active_coords.int(),
                feats=torch.randn(min_active_coords.shape[0], latent_dim, device=device),
            )
            z_pred = sampler.sample(
                model=vflow,
                noise=noise,
                cond=cond,
                neg_cond=neg_cond,
                steps=args.steps,
                cfg_strength=args.cfg_strength,
                rescale_t=args.rescale_t,
                density=density,
            )

            pred_coords, pred_offsets = decode_vertices(
                vvae, offset_head, z_pred, args.inference_threshold
            )
            if args.reconstruct:
                recon_coords, recon_offsets = decode_vertices(
                    vvae, offset_head, z_gt, args.inference_threshold
                )

        gt_vox = batch[f"gt_vertex_voxels_{res}"]
        gt_off = batch[f"gt_vertex_offsets_{res}"]
        for b, name in enumerate(batch["name"]):
            gt_sel = gt_vox[:, 0] == b
            export_vertex(
                args.out_dir,
                name,
                type_name="gt",
                vert_int=gt_vox[gt_sel, 1:].numpy(),
                vert_offsets=gt_off[gt_sel].numpy(),
                resolution=res,
            )
            if args.reconstruct:
                rec_sel = recon_coords[:, 0] == b
                export_vertex(
                    args.out_dir,
                    name,
                    type_name="recon",
                    vert_int=recon_coords[rec_sel, 1:].numpy(),
                    vert_offsets=recon_offsets[rec_sel].numpy(),
                    resolution=res,
                )
            pred_sel = pred_coords[:, 0] == b
            export_vertex(
                args.out_dir,
                name,
                type_name="pred",
                vert_int=pred_coords[pred_sel, 1:].numpy(),
                vert_offsets=pred_offsets[pred_sel].numpy(),
                resolution=res,
            )
            Image.fromarray(batch["image"][b]).save(
                os.path.join(args.out_dir, f"{name}_render.png")
            )

            n_ok += 1

    logging.info(
        f"done: {n_ok} ok, {n_fail} failed in {time.time() - t_start:.0f}s -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
