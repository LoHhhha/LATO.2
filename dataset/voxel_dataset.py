import os
from typing import Dict, List, Optional
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from dataset.utils import (
    MESH_EXTENSIONS,
    dedup_quantized_mesh,
    extract_active_voxels,
    quantize_mesh_clustering,
    sample_point_features,
    union_voxels,
)


class VoxelVertexDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        resolution: int = 1024,
        min_resolution: int = 64,
        pc_sample_number: int = 819200,
        sample_type: str = "dora",
        normalize_vdf: bool = True,
        need_encoder_inputs: bool = True,
        min_vertices: int = 0,
        max_vertices: Optional[int] = None,
        num_samples: Optional[int] = None,
        render: bool = False,
        img_res: int = 518,
        render_azimuth: float = 45.0,
        render_elevation: float = 30.0,
    ):
        self.root_dir = root_dir
        self.resolution = resolution
        self.min_resolution = min_resolution
        self.pc_sample_number = pc_sample_number
        self.sample_type = sample_type
        self.normalize_vdf = normalize_vdf
        self.need_encoder_inputs = need_encoder_inputs
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices
        self.render = render
        self.img_res = img_res
        self.render_azimuth = render_azimuth
        self.render_elevation = render_elevation

        self.files = sorted(
            f
            for f in os.listdir(root_dir)
            if os.path.splitext(f)[1].lower() in MESH_EXTENSIONS
        )
        if num_samples is not None:
            self.files = self.files[:num_samples]
        if not self.files:
            raise ValueError(f"no mesh files ({MESH_EXTENSIONS}) under {root_dir}")

        self._renderer = None  # lazy: one EGL context per DataLoader worker

    def __len__(self) -> int:
        return len(self.files)

    def _render_image(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        if self._renderer is None:
            from dataset.mesh_render import WhiteModelRenderer

            self._renderer = WhiteModelRenderer(
                img_res=self.img_res,
                mesh_color=(0.78, 0.78, 0.82),
                bg_color=(0.0, 0.0, 0.0),
                up_axis="y",
                add_ground=False,
                shadow=True,
                crop_to_object=True,
                crop_padding=1.2,
            )
        imgs, _ = self._renderer.render(
            np.asarray(vertices, dtype=np.float64),
            np.asarray(faces, dtype=np.int64),
            num_views=1,
            azimuths=[self.render_azimuth],
            elevations=[self.render_elevation],
        )
        return imgs[0]  # (img_res, img_res, 3) uint8

    def __getitem__(self, idx: int) -> Dict:
        name = os.path.splitext(self.files[idx])[0]
        path = os.path.join(self.root_dir, self.files[idx])
        res = self.resolution
        min_res = self.min_resolution
        try:
            quantized = quantize_mesh_clustering(path, resolution=res)
            if quantized is None:
                return {"name": name, "error": "empty mesh"}
            v_int, offsets, faces = quantized
            if len(faces) < 1 or len(v_int) < 3:
                return {"name": name, "error": "degenerate mesh after quantization"}

            gt_int, gt_offsets, gt_faces = dedup_quantized_mesh(
                v_int, offsets, faces, res
            )
            num_gt = len(gt_int)
            if num_gt < max(self.min_vertices, 3) or len(gt_faces) < 1:
                return {
                    "name": name,
                    "error": f"too few vertices/faces after dedup ({num_gt})",
                }
            if self.max_vertices is not None and num_gt > self.max_vertices:
                return {
                    "name": name,
                    "error": f"vertex count {num_gt} exceeds max_vertices={self.max_vertices}",
                }

            quant_v = gt_int.astype(np.float64) / (res - 1.0) - 0.5
            quant_v = np.clip(quant_v, -0.5 + 1e-6, 0.5 - 1e-6).astype(np.float32)
            tmesh = trimesh.Trimesh(vertices=quant_v, faces=gt_faces, process=False)

            if self.need_encoder_inputs:
                vertex_added_active = extract_active_voxels(quant_v, gt_faces, res)
                vertex_added_active = union_voxels(
                    vertex_added_active, torch.from_numpy(gt_int), res
                )
                point_cloud = sample_point_features(
                    tmesh,
                    self.pc_sample_number,
                    sample_type=self.sample_type,
                    normalize_vdf=self.normalize_vdf,
                )
            else:
                vertex_added_active = torch.zeros((0, 3), dtype=torch.int32)
                point_cloud = torch.zeros((0, 15), dtype=torch.float32)

            min_active = extract_active_voxels(quant_v, gt_faces, self.min_resolution)

            data = {
                "name": name,
                f"gt_vertex_voxels_{res}": torch.from_numpy(gt_int),
                f"gt_vertex_offsets_{res}": torch.from_numpy(gt_offsets),
                "quantized_vertices": torch.from_numpy(quant_v),
                "quantized_faces": torch.from_numpy(gt_faces),
                f"vertex_added_active_voxels_{res}": vertex_added_active,
                f"point_cloud_{res}": point_cloud,
                f"active_voxels_{min_res}": min_active,
            }

            # Raw mesh, bbox-normalized into the same [-0.5, 0.5] frame.
            raw = trimesh.load(path, process=False, force="mesh")
            raw_v = np.asarray(raw.vertices, dtype=np.float64)
            center = (raw_v.min(axis=0) + raw_v.max(axis=0)) / 2.0
            extent = max(float((raw_v.max(axis=0) - raw_v.min(axis=0)).max()), 1e-7)
            data["original_vertices"] = torch.from_numpy(
                ((raw_v - center) / extent).astype(np.float32)
            )
            data["original_faces"] = torch.from_numpy(
                np.asarray(raw.faces, dtype=np.int64)
            )

            if self.render:
                render_v = v_int.astype(np.float64) / res - 0.5
                data["image"] = self._render_image(render_v, faces)

            return data
        except Exception as e:
            import traceback

            return {"name": name, "error": f"{e}\n{traceback.format_exc()}"}


def collate_fn(
    batch: List[Dict], resolution: int = 1024, min_resolution: int = 64
) -> Dict:
    res = resolution
    min_res = min_resolution
    errors = [b for b in batch if "error" in b]
    batch = [b for b in batch if "error" not in b]
    collated: Dict = {"errors": errors}
    if not batch:
        return collated

    collated["name"] = [b["name"] for b in batch]
    for key in (
        "quantized_vertices",
        "quantized_faces",
        "original_vertices",
        "original_faces",
    ):
        collated[key] = [b[key] for b in batch]
    if "image" in batch[0]:
        collated["image"] = [b["image"] for b in batch]

    for key in (
        f"gt_vertex_voxels_{res}",
        f"vertex_added_active_voxels_{res}",
        f"active_voxels_{min_res}",
    ):
        rows = []
        for i, b in enumerate(batch):
            coords = b[key]
            batch_idx = torch.full((coords.shape[0], 1), i, dtype=torch.int32)
            rows.append(torch.cat([batch_idx, coords], dim=1))
        collated[key] = torch.cat(rows, dim=0)

    collated[f"gt_vertex_offsets_{res}"] = torch.cat(
        [b[f"gt_vertex_offsets_{res}"] for b in batch], dim=0
    )
    collated[f"point_cloud_{res}"] = torch.stack(
        [b[f"point_cloud_{res}"] for b in batch], dim=0
    )
    return collated
