"""
Outputs PLYs into --out_dir:
    <mesh_id>_gt_coords.ply     GT vertex voxel coords in [0, 1024)
    <mesh_id>_gt.ply            GT vertices (+GT offsets) in [-0.5, 0.5]
    <mesh_id>_recon_coords.ply  reconstructed vertex voxel coords in [0, 1024)
    <mesh_id>_recon.ply         reconstructed vertices (+offset head) in [-0.5, 0.5]

Usage:
    python scripts/vvae_inference.py --mesh_dir <dir> --out_dir outputs/vvae_run/<dir> \
        [--batch_size 2] [--num_samples 8]
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

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.voxel_dataset import VoxelVertexDataset, collate_fn
from models import OffsetHead, VDFEncoder, VertexVAE
from modules.sparse import SparseTensor
import utils.logging as logging
from utils.export import export_vertex
from utils.load import load_latov2_model


def parse_args():
    p = argparse.ArgumentParser(description="V-VAE vertex reconstruction inference")
    p.add_argument("--mesh_dir", required=True, help="directory of input meshes")
    p.add_argument("--out_dir", required=True, help="output directory for the PLYs")
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
        "--sample_posterior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sample the posterior (training-style) instead of taking its mode",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.num_samples is not None and args.num_samples <= 0:
        args.num_samples = None  # <= 0 means "all", not python slice semantics
    return args


def main():
    logging.info("V-VAE inference starting...")

    args = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    vvae, vvae_cfg = load_latov2_model(VertexVAE, args.vvae_ckpt, device)
    vdf_encoder, _ = load_latov2_model(VDFEncoder, args.vdf_encoder_ckpt, device)
    offset_head, _ = load_latov2_model(OffsetHead, args.offset_head_ckpt, device)
    res = vvae_cfg["resolution"]

    dataset = VoxelVertexDataset(
        root_dir=args.mesh_dir,
        resolution=res,
        pc_sample_number=args.pc_sample_number,
        sample_type=args.sample_type,
        num_samples=args.num_samples,
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, resolution=res),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logging.info(
        f"{len(dataset)} meshes from {args.mesh_dir} "
        f"(batch_size={args.batch_size}, seed={args.seed}, "
        f"sample_posterior={args.sample_posterior}) -> {args.out_dir}"
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

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            point_cloud = batch[f"point_cloud_{res}"].to(device)
            vertex_added_active_coords = batch[f"vertex_added_active_voxels_{res}"].to(device)

            feats = vdf_encoder(
                p=point_cloud,
                sparse_coords=vertex_added_active_coords,
                res=res,
                bbox_size=(-0.5, 0.5),
            )
            z, _ = vvae.encode(
                SparseTensor(feats=feats, coords=vertex_added_active_coords.int()),
                sample_posterior=args.sample_posterior,
            )
            decoded = vvae.decode(
                z,
                gt_vertex_voxels_list=[],
                training=False,
                inference_threshold=args.inference_threshold,
            )
            recon_coords = decoded[-1]["coords"]  # (N, 4) int, batch col first
            recon_offsets = offset_head(
                decoded[-1]["feats"]
            ).float()  # (N, 3) in (-1, 1)

        gt_vox = batch[f"gt_vertex_voxels_{res}"]
        gt_off = batch[f"gt_vertex_offsets_{res}"]
        recon_coords = recon_coords.cpu()
        recon_offsets = recon_offsets.cpu()
        for b, name in enumerate(batch["name"]):
            gt_sel = gt_vox[:, 0] == b
            rec_sel = recon_coords[:, 0] == b
            export_vertex(
                args.out_dir,
                name,
                type_name="gt",
                vert_int=gt_vox[gt_sel, 1:].numpy(),
                vert_offsets=gt_off[gt_sel].numpy(),
                resolution=res,
            )
            export_vertex(
                args.out_dir,
                name,
                type_name="recon",
                vert_int=recon_coords[rec_sel, 1:].numpy(),
                vert_offsets=recon_offsets[rec_sel].numpy(),
                resolution=res,
            )
            n_ok += 1

    logging.info(
        f"done: {n_ok} ok, {n_fail} failed in {time.time() - t_start:.0f}s -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
