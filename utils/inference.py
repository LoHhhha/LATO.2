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


def _shortest_open_ring(
    u: int, v: int, open_nbr: list[set], nbr: list[set], max_size: int
) -> list[int] | None:
    # Shortest cycle of 4..max_size vertices through the open edge (u, v) that
    # only walks open edges. Vertices adjacent to both u and v would put a chord
    # on any such cycle, so they are excluded up front.
    blocked = nbr[u] & nbr[v]
    parent = {v: -1}
    frontier = [v]
    for _ in range(max_size - 1):
        reached = []
        for x in frontier:
            for y in open_nbr[x]:
                if y in parent or y in blocked:
                    continue
                if y == u:
                    if x == v:
                        continue  # the edge (u, v) itself
                    ring = [u]
                    while x != -1:
                        ring.append(x)
                        x = parent[x]
                    return ring
                parent[y] = x
                reached.append(y)
        if not reached:
            break
        frontier = reached
    return None


def _is_chordless(ring: list[int], nbr: list[set]) -> bool:
    n = len(ring)
    for i in range(n - 2):
        nbr_i = nbr[ring[i]]
        # ring[0] and ring[-1] are ring-adjacent, not a chord
        for j in range(i + 2, n - 1 if i == 0 else n):
            if ring[j] in nbr_i:
                return False
    return True


def _min_area_triangulation(ring: list[int], verts: np.ndarray) -> list[list[int]]:
    # Minimum-area triangulation of a closed 3D vertex ring (O(n^3) DP). The cost
    # is (number of zero-area triangles, total area) so that collinear ring
    # vertices do not yield degenerate faces unless unavoidable.
    n = len(ring)
    p = np.asarray(verts, dtype=np.float64)[ring]
    d = p[None, :, :] - p[:, None, :]  # d[i, k] = p[k] - p[i]
    area = 0.5 * np.linalg.norm(
        np.cross(d[:, :, None, :], d[:, None, :, :]), axis=-1
    )  # area[i, k, j] of triangle (i, k, j)
    eps = 1e-9 * float(np.sum(np.ptp(p, axis=0) ** 2))
    area = area.tolist()
    cost = [[(0, 0.0)] * n for _ in range(n)]
    split = [[-1] * n for _ in range(n)]
    for span in range(2, n):
        for i in range(n - span):
            j = i + span
            best = None
            for k in range(i + 1, j):
                a = area[i][k][j]
                c = (
                    cost[i][k][0] + cost[k][j][0] + (a <= eps),
                    cost[i][k][1] + cost[k][j][1] + a,
                )
                if best is None or c < best:
                    best = c
                    split[i][j] = k
            cost[i][j] = best
    tris = []
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        k = split[i][j]
        tris.append([ring[i], ring[k], ring[j]])
        stack.append((i, k))
        stack.append((k, j))
    return tris


def triangulate_open_rings(
    adj: np.ndarray, faces: np.ndarray, verts: np.ndarray, max_ring_size: int = 12
) -> np.ndarray:
    # Hole filling on open edges, i.e. predicted edges with fewer than two
    # incident faces (boundary edges and face-less wire edges). Chordless rings
    # of 4..max_ring_size open edges are triangulated, smallest rings first, and
    # face counts are updated after every fill so no edge is closed twice.
    empty = np.empty((0, 3), dtype=np.int32)
    num = int(adj.shape[0])
    if num < 4 or max_ring_size < 4:
        return empty
    nbr = [set(np.nonzero(adj[v])[0].tolist()) for v in range(num)]
    count: dict[tuple[int, int], int] = {}
    for a, b, c in faces.tolist():
        for x, y in ((a, b), (b, c), (c, a)):
            key = (x, y) if x < y else (y, x)
            count[key] = count.get(key, 0) + 1
    open_nbr: list[set] = [set() for _ in range(num)]
    open_edges = []
    for u in range(num):
        for v in sorted(nbr[u]):
            if v > u and count.get((u, v), 0) < 2:
                open_edges.append((u, v))
                open_nbr[u].add(v)
                open_nbr[v].add(u)

    new_faces: list[list[int]] = []
    for size in range(4, max_ring_size + 1):
        for u, v in open_edges:
            # a face-less edge can border two rings, hence the loop
            while count.get((u, v), 0) < 2:
                ring = _shortest_open_ring(u, v, open_nbr, nbr, size)
                if ring is None or not _is_chordless(ring, nbr):
                    break
                for tri in _min_area_triangulation(ring, verts):
                    new_faces.append(tri)
                    for x, y in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                        key = (x, y) if x < y else (y, x)
                        count[key] = count.get(key, 0) + 1
                        nbr[x].add(y)
                        nbr[y].add(x)
                        if count[key] >= 2:
                            open_nbr[x].discard(y)
                            open_nbr[y].discard(x)
    if not new_faces:
        return empty
    return np.asarray(new_faces, dtype=np.int32)


def edges_to_faces(
    edges: np.ndarray,
    num_valid: int,
    verts: np.ndarray,
    fill_open_rings: bool = True,
    max_ring_size: int = 12,
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
    if fill_open_rings:
        fill = triangulate_open_rings(adj, faces, verts, max_ring_size)
        if fill.shape[0] > 0:
            faces = np.concatenate([faces, fill], axis=0) if faces.shape[0] else fill
    return faces
