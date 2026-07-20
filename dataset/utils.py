import numpy as np
import torch
import trimesh
from trimesh import grouping

from o_voxel.convert import mesh_to_flexible_dual_grid

MESH_EXTENSIONS = {".obj", ".glb", ".gltf", ".ply", ".stl", ".off"}


def quantize_mesh_clustering(mesh_path: str, resolution: int = 1024):
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    bbox_min, bbox_max = vertices.min(axis=0), vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    max_extent = max(float((bbox_max - bbox_min).max()), 1e-7)
    normalized_v = (vertices - center) / max_extent + 0.5  # [0, 1]

    v_grid = np.clip(np.floor(normalized_v * resolution), 0, resolution - 1).astype(
        np.int64
    )
    v_hash = (
        v_grid[:, 0] * resolution * resolution
        + v_grid[:, 1] * resolution
        + v_grid[:, 2]
    )

    unique_hashes, inverse = np.unique(v_hash, return_inverse=True)
    num_clusters = len(unique_hashes)

    v_sum = np.zeros((num_clusters, 3), dtype=np.float64)
    counts = np.zeros(num_clusters, dtype=np.float64)
    np.add.at(v_sum, inverse, normalized_v)
    np.add.at(counts, inverse, 1)
    v_mean = v_sum / counts[:, None]

    v_int = np.clip(np.floor(v_mean * resolution), 0, resolution - 1).astype(np.int32)

    # Offset of the cluster mean relative to the voxel center, scaled to (-1, 1).
    voxel_center = (v_int.astype(np.float64) + 0.5) / float(resolution)
    offsets = ((v_mean - voxel_center) * 2.0 * resolution).astype(np.float32)
    offsets = np.clip(offsets, -1.0, 1.0)

    new_faces = inverse[faces]
    valid = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 2] != new_faces[:, 0])
    )
    return v_int, offsets, new_faces[valid]


def realign_offsets(
    orig_int: np.ndarray, orig_off: np.ndarray, kept_int: np.ndarray, resolution: int
) -> np.ndarray:
    base = resolution * resolution
    orig_i = orig_int.astype(np.int64)
    kept_i = kept_int.astype(np.int64)
    orig_h = orig_i[:, 0] * base + orig_i[:, 1] * resolution + orig_i[:, 2]
    kept_h = kept_i[:, 0] * base + kept_i[:, 1] * resolution + kept_i[:, 2]

    order = np.argsort(orig_h)
    sorted_h = orig_h[order]
    sorted_off = orig_off.astype(np.float64)[order]
    cumsum = np.concatenate([np.zeros((1, 3)), np.cumsum(sorted_off, axis=0)], axis=0)
    left = np.searchsorted(sorted_h, kept_h, side="left")
    right = np.searchsorted(sorted_h, kept_h, side="right")
    counts = (right - left).clip(min=1).astype(np.float64)
    out = ((cumsum[right] - cumsum[left]) / counts[:, None]).astype(np.float32)
    out = np.clip(out, -1.0, 1.0)
    out[right <= left] = 0.0
    return out


def dedup_quantized_mesh(
    v_int: np.ndarray, offsets: np.ndarray, faces: np.ndarray, resolution: int
):
    tmesh = trimesh.Trimesh(vertices=v_int, faces=faces, process=False)
    tmesh.merge_vertices()
    tmesh.update_faces(tmesh.nondegenerate_faces())
    tmesh.update_faces(tmesh.unique_faces())
    tmesh.remove_unreferenced_vertices()

    gt_int = np.asarray(tmesh.vertices).astype(np.int32)
    gt_faces = np.asarray(tmesh.faces, dtype=np.int64)
    gt_offsets = realign_offsets(v_int, offsets, gt_int, resolution)
    return gt_int, gt_offsets, gt_faces


def extract_active_voxels(
    vertices: np.ndarray, faces: np.ndarray, resolution: int
) -> torch.Tensor:
    coords, *_ = mesh_to_flexible_dual_grid(
        vertices=torch.as_tensor(vertices * 0.99999, dtype=torch.float32).contiguous(),
        faces=torch.as_tensor(faces, dtype=torch.int32).contiguous(),
        grid_size=resolution,
        aabb=torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32),
    )
    coords = coords.cpu().long()
    coords_1d = (
        coords[:, 0] * resolution * resolution
        + coords[:, 1] * resolution
        + coords[:, 2]
    )
    return coords[torch.argsort(coords_1d)].int()


def union_voxels(a: torch.Tensor, b: torch.Tensor, resolution: int) -> torch.Tensor:
    if a.numel() == 0:
        return b.int().clone()
    if b.numel() == 0:
        return a.int().clone()
    combined = torch.cat([a.reshape(-1, 3).long(), b.reshape(-1, 3).long()], dim=0)
    combined = combined.clamp(0, resolution - 1)
    hashes = (
        combined[:, 0] * resolution * resolution
        + combined[:, 1] * resolution
        + combined[:, 2]
    )
    sorted_hashes, sort_idx = torch.sort(hashes)
    keep = torch.ones_like(sorted_hashes, dtype=torch.bool)
    keep[1:] = sorted_hashes[1:] != sorted_hashes[:-1]
    return combined[sort_idx[keep]].int()


def _sample_surface_uniform(tm_mesh: trimesh.Trimesh, n_samples: int):
    face_idx = np.random.choice(len(tm_mesh.faces), size=n_samples, replace=True)
    tri = tm_mesh.vertices[tm_mesh.faces[face_idx]]  # (N, 3, 3)
    u = np.random.rand(n_samples, 1)
    v = np.random.rand(n_samples, 1)
    sqrt_u = np.sqrt(u)
    points = (
        (1 - sqrt_u) * tri[:, 0]
        + (sqrt_u * (1 - v)) * tri[:, 1]
        + (sqrt_u * v) * tri[:, 2]
    )
    normals = tm_mesh.face_normals[face_idx]
    return points.astype(np.float32), normals.astype(np.float32), face_idx


def _sample_edges_dora(
    tm_mesh: trimesh.Trimesh, n_len_samples: int, n_uniform_samples: int
):
    parts_start, parts_end, parts_norm, parts_virt = [], [], [], []

    adj_faces = tm_mesh.face_adjacency
    adj_edges = tm_mesh.face_adjacency_edges
    if len(adj_faces) > 0:
        n0 = tm_mesh.face_normals[adj_faces[:, 0]]
        n1 = tm_mesh.face_normals[adj_faces[:, 1]]
        sum_normals = n0 + n1
        norms = np.linalg.norm(sum_normals, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0

        faces_pair = tm_mesh.faces[adj_faces]
        unique_idx_0 = np.sum(faces_pair, axis=2)[:, 0] - np.sum(adj_edges, axis=1)
        unique_idx_1 = np.sum(faces_pair, axis=2)[:, 1] - np.sum(adj_edges, axis=1)
        virtual = (
            tm_mesh.vertices[unique_idx_0] + tm_mesh.vertices[unique_idx_1]
        ) * 0.5

        parts_start.append(tm_mesh.vertices[adj_edges[:, 0]])
        parts_end.append(tm_mesh.vertices[adj_edges[:, 1]])
        parts_norm.append(sum_normals / norms)
        parts_virt.append(virtual)

    edges_sorted = tm_mesh.edges_sorted
    if len(edges_sorted) > 0:
        boundary_group = grouping.group_rows(edges_sorted, require_count=1)
        if len(boundary_group) > 0:
            boundary_indices = np.concatenate(
                [np.atleast_1d(g) for g in boundary_group]
            )
            face_indices = boundary_indices // 3
            edge_v = edges_sorted[boundary_indices]
            unique_idx = np.sum(tm_mesh.faces[face_indices], axis=1) - np.sum(
                edge_v, axis=1
            )

            parts_start.append(tm_mesh.vertices[edge_v[:, 0]])
            parts_end.append(tm_mesh.vertices[edge_v[:, 1]])
            parts_norm.append(tm_mesh.face_normals[face_indices])
            parts_virt.append(tm_mesh.vertices[unique_idx])

    if not parts_start:
        return None, None, None

    v_start = np.concatenate(parts_start, axis=0)
    v_end = np.concatenate(parts_end, axis=0)
    normals = np.concatenate(parts_norm, axis=0)
    v_virtual = np.concatenate(parts_virt, axis=0)

    lengths = np.linalg.norm(v_end - v_start, axis=1)
    total = lengths.sum()
    num_edges = len(lengths)
    probs_len = (
        lengths / total if total >= 1e-9 else np.full(num_edges, 1.0 / num_edges)
    )
    probs_len = probs_len / probs_len.sum()

    chosen = np.concatenate(
        [
            np.random.choice(num_edges, size=n_len_samples, p=probs_len),
            np.random.choice(num_edges, size=n_uniform_samples),
        ]
    )
    t = np.random.rand(len(chosen), 1)
    points = v_start[chosen] + (v_end[chosen] - v_start[chosen]) * t
    triplets = np.stack([v_start[chosen], v_end[chosen], v_virtual[chosen]], axis=1)
    return (
        points.astype(np.float32),
        normals[chosen].astype(np.float32),
        triplets.astype(np.float32),
    )


def _vdf_from_triplets(
    points: np.ndarray, triplets: np.ndarray, normalize: bool
) -> np.ndarray:
    view_dtype = np.dtype((np.void, triplets.dtype.itemsize * triplets.shape[-1]))
    v_view = triplets.view(view_dtype).squeeze(-1)
    sort_idx = np.argsort(v_view, axis=1)
    v_sorted = triplets[np.arange(triplets.shape[0])[:, None], sort_idx]

    dirs = v_sorted - points[:, None, :]  # (N, 3, 3)
    if normalize:
        dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8)
    return dirs.reshape(len(points), 9).astype(np.float32)


def sample_point_features(
    tm_mesh: trimesh.Trimesh,
    n_samples: int,
    sample_type: str = "dora",
    normalize_vdf: bool = True,
) -> torch.Tensor:
    # (N, 15) float32 point features: [xyz(3), normal(3), vdf(9)].
    vertices = np.asarray(tm_mesh.vertices, dtype=np.float64)
    faces = np.asarray(tm_mesh.faces)

    if sample_type == "dora":
        n_surf_area = n_samples // 4
        n_surf_uniform = n_samples // 4
        n_edge_len = n_samples // 4
        n_edge_uniform = n_samples - n_surf_area - n_surf_uniform - n_edge_len

        p_edge, n_edge, triplets_edge = _sample_edges_dora(
            tm_mesh, n_edge_len, n_edge_uniform
        )
        if p_edge is None:
            n_surf_area += n_edge_len
            n_surf_uniform += n_edge_uniform
    elif sample_type == "uniform":
        n_surf_area, n_surf_uniform = n_samples, 0
        p_edge = None
    else:
        raise ValueError(f"unknown sample_type: {sample_type!r}")

    p_area, idx_area = tm_mesh.sample(n_surf_area, return_index=True)
    n_area = tm_mesh.face_normals[idx_area]
    if n_surf_uniform > 0:
        p_unif, n_unif, idx_unif = _sample_surface_uniform(tm_mesh, n_surf_uniform)
        points = np.concatenate([p_area, p_unif], axis=0).astype(np.float32)
        normals = np.concatenate([n_area, n_unif], axis=0).astype(np.float32)
        idx_surf = np.concatenate([idx_area, idx_unif], axis=0)
    else:
        points = p_area.astype(np.float32)
        normals = n_area.astype(np.float32)
        idx_surf = idx_area
    triplets = vertices[faces[idx_surf]].astype(np.float32)

    if p_edge is not None:
        points = np.concatenate([points, p_edge], axis=0)
        normals = np.concatenate([normals, n_edge], axis=0)
        triplets = np.concatenate([triplets, triplets_edge], axis=0)

    vdf = _vdf_from_triplets(points, triplets, normalize=normalize_vdf)
    return torch.from_numpy(np.concatenate([points, normals, vdf], axis=-1))
