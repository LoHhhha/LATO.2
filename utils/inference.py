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


def _fibonacci_dirs(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    phi = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def _face_visibility(tri, normal, area, num_dirs=48, res=160, samples=80000, seed=0):
    """Per face: summed n.d over view directions it is seen from, split by sign.

    Each direction gets an orthographic depth buffer splatted from surface samples
    and max-filtered over 3x3 cells so occluders have no pinholes; a face is seen
    when its centroid is at the front of its cell. Needs no orientation and
    tolerates holes. Returns (front, back): seen with n.d > 0 and with n.d < 0.
    """
    rng = np.random.default_rng(seed)
    fi = rng.choice(len(tri), size=samples, p=area / max(area.sum(), 1e-12))
    uv = rng.random((samples, 2))
    over = uv.sum(1) > 1
    uv[over] = 1 - uv[over]
    t = tri[fi]
    pts = t[:, 0] + uv[:, :1] * (t[:, 1] - t[:, 0]) + uv[:, 1:] * (t[:, 2] - t[:, 0])
    cen = tri.mean(1)
    pix = 1.2 / res
    front = np.zeros(len(tri))
    back = np.zeros(len(tri))
    for d in _fibonacci_dirs(num_dirs):
        a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(d, a)
        u /= np.linalg.norm(u)
        w = np.cross(d, u)

        def cell(p):
            ij = np.floor((np.stack([p @ u, p @ w], 1) + 0.6) / pix).astype(np.int64)
            return np.clip(ij, 0, res - 1) @ np.array([res, 1])

        depth = np.full(res * res, -np.inf)
        np.maximum.at(depth, cell(pts), pts @ d)
        grid = np.pad(depth.reshape(res, res), 1, constant_values=-np.inf)
        depth = np.max([grid[i:i + res, j:j + res] for i in range(3) for j in range(3)], axis=0).ravel()
        nd = normal @ d
        # A slanted face's own samples sit higher in neighbouring cells; allow for its slope.
        slope = np.sqrt(np.maximum(1 - nd * nd, 0)) / np.maximum(np.abs(nd), 0.2)
        seen = cen @ d >= depth[cell(cen)] - pix * (1.0 + 0.5 * slope)
        front += seen * np.maximum(nd, 0)
        back += seen * np.maximum(-nd, 0)
    return front, back


def orient_faces(
    faces: np.ndarray,
    verts: np.ndarray,
    num_dirs: int = 48,
    conflict: float = 0.001,
    seam_cos: float = 0.9,
) -> np.ndarray:
    """Consistent, outward winding for a triangle soup whose index order is arbitrary.

    Faces are joined across two-face edges from the smoothest edge to the sharpest,
    tracking which faces must flip relative to each other. An edge is left as a seam
    when it would contradict earlier joins (non-orientable surface) or when both
    sides already have clear, opposite evidence of which way they face; so seams
    fall on sharp folds rather than wherever a traversal happens to close a loop.
    Each joined patch is then flipped as a whole so its faces point toward the view
    directions they are visible from (signed volume if never visible). Only the
    winding changes; the set of triangles is the same.
    """
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3).copy()
    num_faces = faces.shape[0]
    v = np.asarray(verts, dtype=np.float64)
    if num_faces == 0:
        return faces.astype(np.int32)
    lo, hi = v.min(0), v.max(0)
    v = (v - (lo + hi) / 2) / max(float((hi - lo).max()), 1e-12)

    tri = v[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    normal = cross / np.maximum(2 * area, 1e-12)[:, None]

    # Two-face edges: which faces they join and whether those faces disagree.
    half = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    owner = np.repeat(np.arange(num_faces), 3)
    key = np.sort(half, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key_sorted = key[order]
    starts = np.flatnonzero(np.r_[True, np.any(key_sorted[1:] != key_sorted[:-1], axis=1)])
    counts = np.diff(np.r_[starts, len(order)])
    pairs = order[starts[counts == 2][:, None] + np.array([0, 1])]
    fa, fb = owner[pairs[:, 0]], owner[pairs[:, 1]]
    disagree = half[pairs[:, 0], 0] == half[pairs[:, 1], 0]
    keep = fa != fb
    fa, fb, disagree = fa[keep], fb[keep], disagree[keep]
    smooth = np.einsum("ij,ij->i", normal[fa], normal[fb]) * np.where(disagree, -1.0, 1.0)

    front, back = _face_visibility(tri, normal, area, num_dirs) if num_dirs else (np.zeros(num_faces),) * 2
    votes = (front - back) * area
    # Patches that both have more than this much evidence and disagree stay apart.
    tau = conflict * np.abs(votes).sum() if conflict >= 0 else np.inf

    # Union-find with parity: rel[f] = flip of f relative to its parent;
    # pvote[root] = visibility vote of the root's current orientation.
    parent = np.arange(num_faces)
    rel = np.zeros(num_faces, dtype=bool)
    pvote = votes.copy()

    def find(f):
        path = []
        while parent[f] != f:
            path.append(f)
            f = parent[f]
        acc = False
        for g in reversed(path):
            acc ^= rel[g]
            rel[g] = acc
            parent[g] = f
        return f

    for e in np.argsort(-smooth, kind="stable"):
        a, b = fa[e], fb[e]
        ra, rb = find(a), find(b)
        if ra != rb:
            q = rel[a] ^ rel[b] ^ disagree[e]
            vb = -pvote[rb] if q else pvote[rb]
            # At a sharp edge whose two sides already know they face opposite ways: keep a seam.
            if smooth[e] < seam_cos and pvote[ra] * vb < 0 and min(abs(pvote[ra]), abs(vb)) > tau:
                continue
            parent[rb] = ra
            rel[rb] = q
            pvote[ra] += vb
    roots = np.array([find(f) for f in range(num_faces)])
    flip = rel.copy()

    # Outward per patch: visibility vote, signed volume when never seen.
    sign = np.where(flip, -1.0, 1.0)
    patch_vote = np.bincount(roots, votes * sign, num_faces)
    cen = np.stack([np.bincount(roots, tri[:, :, d].mean(1) * area, num_faces) for d in range(3)], 1)
    cen /= np.maximum(np.bincount(roots, area, num_faces), 1e-12)[:, None]
    rel_tri = tri - cen[roots][:, None, :]
    six_vol = np.einsum("ij,ij->i", rel_tri[:, 0], np.cross(rel_tri[:, 1], rel_tri[:, 2])) * sign
    patch_vol = np.bincount(roots, six_vol, num_faces)
    seen = np.abs(patch_vote) > 1e-9
    inward = np.where(seen, patch_vote < 0, patch_vol < 0)
    flip ^= inward[roots]
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces.astype(np.int32)


def edges_to_faces(
    edges: np.ndarray,
    num_valid: int,
    verts: np.ndarray,
    fill_open_rings: bool = True,
    max_ring_size: int = 12,
    orient: bool = True,
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
    # Triangles come out of the edge cliques with arbitrary winding (u < v < w).
    if orient:
        faces = orient_faces(faces, verts)
    return faces
