from __future__ import annotations

import os
import traceback
from typing import Dict, List, Optional

import numpy as np
import torch

from dataset.utils import (
    MESH_EXTENSIONS,
    dedup_quantized_mesh,
    extract_active_voxels,
    quantize_mesh_clustering,
)


class TopoVoxelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        num_discrete: int = 1024,
        voxel_res: int = 64,
        max_vertices: Optional[int] = None,
        num_samples: Optional[int] = None,
    ):
        self.root_dir = root_dir
        self.num_discrete = int(num_discrete)
        self.voxel_res = int(voxel_res)
        self.max_vertices = max_vertices
        self.files = sorted(
            f
            for f in os.listdir(root_dir)
            if os.path.splitext(f)[1].lower() in MESH_EXTENSIONS
        )
        if num_samples is not None:
            self.files = self.files[:num_samples]
        if not self.files:
            raise ValueError(f"no mesh files ({MESH_EXTENSIONS}) under {root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        name = os.path.splitext(self.files[idx])[0]
        path = os.path.join(self.root_dir, self.files[idx])
        res = self.num_discrete
        try:
            quantized = quantize_mesh_clustering(path, resolution=res)
            if quantized is None:
                return {"name": name, "error": "empty mesh"}
            v_int, offsets, faces = quantized
            if len(faces) < 1 or len(v_int) < 3:
                return {"name": name, "error": "degenerate mesh after quantization"}

            gt_int, _, gt_faces = dedup_quantized_mesh(v_int, offsets, faces, res)
            num_gt = len(gt_int)
            if num_gt < 3 or len(gt_faces) < 1:
                return {"name": name, "error": f"too few vertices/faces ({num_gt})"}
            if self.max_vertices is not None and num_gt > self.max_vertices:
                return {
                    "name": name,
                    "error": f"vertex count {num_gt} exceeds max_vertices={self.max_vertices}",
                }

            quant_v = gt_int.astype(np.float64) / (res - 1.0) - 0.5
            quant_v = np.clip(quant_v, -0.5 + 1e-6, 0.5 - 1e-6).astype(np.float32)
            voxel_coords = extract_active_voxels(quant_v, gt_faces, self.voxel_res)

            return {
                "name": name,
                "vertices": torch.from_numpy(gt_int.astype(np.int64)),
                "voxel_coords": voxel_coords.long(),
            }
        except Exception as e:
            return {"name": name, "error": f"{e}\n{traceback.format_exc()}"}


def collate_fn(batch: List[Dict]) -> Dict:
    errors = [b for b in batch if "error" in b]
    good = [b for b in batch if "error" not in b]
    collated: Dict = {"errors": errors}
    if not good:
        return collated
    collated["name"] = [b["name"] for b in good]
    collated["vertices"] = [b["vertices"] for b in good]
    collated["voxel_coords"] = [b["voxel_coords"] for b in good]
    return collated
