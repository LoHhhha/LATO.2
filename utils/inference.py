import os
import numpy as np
import torch


def worker_init(_worker_id):
    import atexit

    atexit.register(os._exit, 0)


def compute_density(batch, args, density_max, device):
    counts = []
    for verts in batch["quantized_vertices"]:
        if args.use_gt_vert_count:
            counts.append(
                min(
                    max(float(verts.shape[0]) * args.scaler, args.min_verts),
                    args.max_verts,
                )
            )
        else:
            counts.append(float(args.vert_num))
    counts = torch.tensor(counts, dtype=torch.float32, device=device)
    return counts / density_max * 1000.0


def decode_vertices(vvae, offset_head, latent, inference_threshold):
    decoded = vvae.decode(
        latent,
        gt_vertex_voxels_list=[],
        training=False,
        inference_threshold=inference_threshold,
    )
    coords = decoded[-1]["coords"]
    offsets = offset_head(decoded[-1]["feats"]).float()
    return coords.cpu(), offsets.cpu()


def build_voxel_fields(voxel_coords_list, voxel_res, device):
    bsz = len(voxel_coords_list)
    field = torch.zeros(
        (bsz, voxel_res, voxel_res, voxel_res), dtype=torch.float32, device=device
    )
    for b, coords in enumerate(voxel_coords_list):
        if coords.numel() > 0:
            c = coords.to(device=device, dtype=torch.long)
            field[b, c[:, 0], c[:, 1], c[:, 2]] = 1.0
    return field


def pad_verts(verts_list, device):
    bsz = len(verts_list)
    lengths = [int(v.shape[0]) for v in verts_list]
    n_max = max(lengths)
    verts = torch.zeros(bsz, n_max, 3, dtype=torch.long, device=device)
    mask = torch.zeros(bsz, n_max, dtype=torch.bool, device=device)
    for b, (v, n) in enumerate(zip(verts_list, lengths)):
        verts[b, :n] = v.to(device=device, dtype=torch.long)
        mask[b, :n] = True
    return verts, mask, lengths


def triangulate_quad_rings(adj: np.ndarray) -> np.ndarray:
    # thinks meshflow@CVPR2026!
    num = int(adj.shape[0])
    if num < 4:
        return np.empty((0, 3), dtype=np.int32)
    nbr_sets = [set(np.nonzero(adj[v])[0].tolist()) for v in range(num)]
    new_faces: list[list[int]] = []
    seen: set[tuple[int, int, int]] = set()
    for a in range(num):
        cand = sorted(v for v in nbr_sets[a] if v > a)
        n_cand = len(cand)
        if n_cand < 2:
            continue
        for i in range(n_cand):
            b = cand[i]
            set_b = nbr_sets[b]
            for j in range(i + 1, n_cand):
                d = cand[j]
                if d in set_b:
                    continue
                for c in set_b & nbr_sets[d]:
                    if c <= a or c in nbr_sets[a]:
                        continue
                    for tri in ([a, b, c], [a, c, d]):
                        key = tuple(sorted(tri))
                        if key in seen:
                            continue
                        seen.add(key)
                        new_faces.append(tri)
    if not new_faces:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(new_faces, dtype=np.int32)


def edges_to_faces(
    edges: np.ndarray, num_valid: int, fill_quad_rings: bool
) -> np.ndarray:
    adj = np.zeros((num_valid, num_valid), dtype=bool)
    if edges.shape[0] > 0:
        adj[edges[:, 0], edges[:, 1]] = True
        adj[edges[:, 1], edges[:, 0]] = True
    faces_list = []
    for ei in range(edges.shape[0]):
        u = int(edges[ei, 0])
        v = int(edges[ei, 1])
        common = np.where(adj[u] & adj[v])[0]
        common = common[common > v]
        for w in common:
            faces_list.append([u, v, int(w)])
    faces = (
        np.asarray(faces_list, dtype=np.int32)
        if faces_list
        else np.empty((0, 3), dtype=np.int32)
    )
    if fill_quad_rings:
        quad = triangulate_quad_rings(adj)
        if quad.shape[0] > 0:
            faces = np.concatenate([faces, quad], axis=0) if faces.shape[0] else quad
    return faces
