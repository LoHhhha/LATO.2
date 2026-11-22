"""CPU fallback for `o_voxel.convert.mesh_to_flexible_dual_grid` (occupied voxels only).

`o_voxel` is built as a CUDA extension and imports nvdiffrast / CuMesh / FlexGEMM,
so it cannot be installed on CPU-only machines or on GPUs without a CUDA toolkit.
The inference scripts only need the *set of active voxels*, i.e. the voxels that
share a grid edge crossed by the surface (same rule as o_voxel's `intersect_qef`).
This module computes that set with NumPy.
"""

from __future__ import annotations

from typing import List

import numpy as np

# Candidate (triangle, grid line) pairs tested per chunk.
_CHUNK = 4_000_000


def _edge_crossings(tri: np.ndarray, axis: int, res: int) -> np.ndarray:
    """Voxels touching the grid edges parallel to `axis` that cross the triangles.

    tri: (T, 3, 3) triangles in grid units ([0, res]).
    """
    a, b = [(axis + 1) % 3, (axis + 2) % 3]
    pu, pv, pd = tri[..., a], tri[..., b], tri[..., axis]

    # Grid lines (u, v) with integer coordinates inside each triangle's bounding box.
    u0 = np.ceil(pu.min(1)).astype(np.int64)
    u1 = np.floor(pu.max(1)).astype(np.int64)
    v0 = np.ceil(pv.min(1)).astype(np.int64)
    v1 = np.floor(pv.max(1)).astype(np.int64)
    nu, nv = np.maximum(u1 - u0 + 1, 0), np.maximum(v1 - v0 + 1, 0)
    count = nu * nv
    keep = np.nonzero(count > 0)[0]
    if keep.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    out: List[np.ndarray] = []
    bounds = np.cumsum(count[keep])
    start = 0
    while start < keep.size:
        base = bounds[start - 1] if start > 0 else 0
        stop = int(np.searchsorted(bounds, base + _CHUNK, side="right"))
        stop = max(stop, start + 1)
        ids = keep[start:stop]
        start = stop

        c = count[ids]
        t = np.repeat(ids, c)
        local = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
        u = (u0[t] + local // nv[t]).astype(np.float64)
        v = (v0[t] + local % nv[t]).astype(np.float64)

        ax, ay, bx, by, cx, cy = pu[t, 0], pv[t, 0], pu[t, 1], pv[t, 1], pu[t, 2], pv[t, 2]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        ok = np.abs(det) > 1e-12
        det = np.where(ok, det, 1.0)
        w0 = ((by - cy) * (u - cx) + (cx - bx) * (v - cy)) / det
        w1 = ((cy - ay) * (u - cx) + (ax - cx) * (v - cy)) / det
        w2 = 1.0 - w0 - w1
        ok &= (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not ok.any():
            continue

        depth = (w0 * pd[t, 0] + w1 * pd[t, 1] + w2 * pd[t, 2])[ok]
        k = np.clip(np.floor(depth).astype(np.int64), 0, res - 1)
        ui, vi = u[ok].astype(np.int64), v[ok].astype(np.int64)
        # The four voxels sharing the crossed grid edge.
        for du in (-1, 0):
            for dv in (-1, 0):
                i, j = ui + du, vi + dv
                inside = (i >= 0) & (i < res) & (j >= 0) & (j < res)
                vox = np.empty((int(inside.sum()), 3), dtype=np.int64)
                vox[:, a], vox[:, b], vox[:, axis] = i[inside], j[inside], k[inside]
                out.append(vox)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3), dtype=np.int64)


def active_voxels(vertices: np.ndarray, faces: np.ndarray, resolution: int) -> np.ndarray:
    """Occupied voxel indices (N, 3) of a mesh whose vertices lie in [-0.5, 0.5]^3."""
    grid = (np.asarray(vertices, dtype=np.float64) + 0.5) * resolution
    tri = grid[np.asarray(faces, dtype=np.int64)]
    vox = np.concatenate([_edge_crossings(tri, axis, resolution) for axis in range(3)], axis=0)
    if vox.shape[0] == 0:
        # Surface smaller than one cell: fall back to the voxels holding the vertices.
        vox = np.clip(np.floor(grid).astype(np.int64), 0, resolution - 1)
    return np.unique(vox, axis=0)
