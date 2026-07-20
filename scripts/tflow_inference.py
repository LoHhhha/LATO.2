"""
Outputs into --out_dir:
    <mesh_id>_pred.obj           generated mesh (known verts, faces) in [-0.5, 0.5]
        <mesh_id>_pred.ply           fallback point cloud when no faces were generated
    <mesh_id>_known.ply          the known (dequantized) vertices fed to the flow
    <mesh_id>_voxel_field.ply    the active-voxel conditioning field (debug, --save_voxel_field)

Usage:
    python scripts/tflow_inference.py --mesh_dir <dir> --out_dir outputs/tflow_run/<dir> \
        [--steps 50] [--no-use_cond] [--no-fill_quad_rings]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import trimesh
import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from torch.utils.data import DataLoader

from dataset.topo_dataset import TopoVoxelDataset, collate_fn
from models import (
    TopologyVAE,
    TopologySiTFlow,
    TopoFlowEulerSampler,
    VoxelFieldConditioner,
)
import utils.logging as logging
from utils.inference import build_voxel_fields, edges_to_faces, pad_verts
from utils.load import load_latov2_model


def parse_args():
    p = argparse.ArgumentParser(description="T-Flow topology generation inference")
    p.add_argument("--mesh_dir", required=True, help="directory of input meshes")
    p.add_argument("--out_dir", required=True, help="output directory for the meshes")
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
    p.add_argument("--seed", type=int, default=42)
    # flow sampling
    p.add_argument("--steps", type=int, default=50, help="Euler steps")
    p.add_argument(
        "--use_cond",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "condition on the active-voxel field. --no-use_cond runs the flow "
            "unconditionally (the model's learned null token)."
        ),
    )
    # topology decoding
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
    p.add_argument(
        "--save_voxel_field",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also dump the active-voxel conditioning field as a point cloud",
    )
    args = p.parse_args()
    if args.num_samples is not None and args.num_samples <= 0:
        args.num_samples = None  # <= 0 means "all", not python slice semantics
    return args


def main():
    logging.info("T-Flow inference starting...")

    args = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    tflow, tflow_cfg = load_latov2_model(TopologySiTFlow, args.tflow_ckpt, device)
    tvae, _ = load_latov2_model(TopologyVAE, args.tvae_ckpt, device)
    voxel_encoder, venc_cfg = load_latov2_model(
        VoxelFieldConditioner, args.voxel_encoder_ckpt, device
    )
    z_dim = int(tflow_cfg["args"]["z_dim"])
    num_discrete = int(tflow_cfg["args"]["num_discrete"])
    max_vertices = int(tflow_cfg["args"]["max_vertices"])
    latent_scale = float(tflow_cfg["latent_scale"])
    voxel_res = int(venc_cfg["resolution"])
    sampler = TopoFlowEulerSampler()

    dataset = TopoVoxelDataset(
        root_dir=args.mesh_dir,
        num_discrete=num_discrete,
        voxel_res=voxel_res,
        max_vertices=max_vertices,
        num_samples=args.num_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logging.info(
        f"{len(dataset)} meshes from {args.mesh_dir} "
        f"(steps={args.steps}, use_cond={args.use_cond}, num_discrete={num_discrete}, "
        f"voxel_res={voxel_res}, latent_scale={latent_scale}, seed={args.seed}) "
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

        names = batch["name"]
        verts_list = batch["vertices"]  # list of (N_i, 3) long in [0, num_discrete)
        voxel_list = batch["voxel_coords"]  # list of (M_i, 3) long in [0, voxel_res)

        with torch.no_grad():
            verts, mask, lengths = pad_verts(
                verts_list, device
            )  # (B, N_max, 3), (B, N_max)
            if args.use_cond:
                field = build_voxel_fields(
                    voxel_list, voxel_res, device
                )  # (B, R, R, R)
                cond = voxel_encoder(field)  # (B, R'^3, cond_in_dim)
            else:
                cond = None  # unconditional

            z0 = torch.randn(verts.shape[0], verts.shape[1], z_dim, device=device)
            z_flow = sampler.sample(
                model=tflow,
                noise=z0,
                verts=verts,
                mask=mask,
                cond=cond,
                steps=args.steps,
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

        for b, name in enumerate(names):
            num_vertices = lengths[b]
            if num_vertices == 0:
                logging.warning(f"{name}: no known vertices; skipping.")
                continue
            edges = edges_list[b]
            faces = edges_to_faces(edges, num_vertices, args.fill_quad_rings)

            verts_int = verts_list[b]
            disp = (verts_int.numpy().astype(np.float64) + 0.5) / num_discrete - 0.5

            if faces.shape[0] > 0:
                trimesh.Trimesh(vertices=disp, faces=faces).export(
                    os.path.join(args.out_dir, f"{name}_pred.obj")
                )
            else:
                trimesh.PointCloud(disp).export(
                    os.path.join(args.out_dir, f"{name}_pred.ply")
                )
            trimesh.PointCloud(disp).export(
                os.path.join(args.out_dir, f"{name}_known.ply")
            )
            if args.use_cond and args.save_voxel_field and voxel_list[b].shape[0] > 0:
                vox_pts = (
                    voxel_list[b].numpy().astype(np.float64) + 0.5
                ) / voxel_res - 0.5
                trimesh.PointCloud(vox_pts).export(
                    os.path.join(args.out_dir, f"{name}_voxel_field.ply")
                )

            n_ok += 1

    logging.info(
        f"done: {n_ok} ok, {n_fail} failed in {time.time() - t_start:.0f}s -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
