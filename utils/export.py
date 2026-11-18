import os
import numpy as np
import trimesh


def export_vertex(out_dir, base_name, type_name, vert_int, vert_offsets, resolution):
    res = float(resolution)
    vert_with_offset = (
        vert_int.astype(np.float64) / res
        - 0.5
        + vert_offsets.astype(np.float64) / (res * 2.0)
    )

    trimesh.PointCloud(vert_with_offset).export(
        os.path.join(out_dir, f"{base_name}_{type_name}.ply")
    )
    trimesh.PointCloud(vert_int).export(
        os.path.join(out_dir, f"{base_name}_{type_name}_coords.ply")
    )


def export_mesh(out_dir, base_name, vert_int, vert_offsets, faces, resolution):
    """OBJ pair matching export_vertex's PLY conventions (offset verts / int coords)."""
    res = float(resolution)
    vert_with_offset = (
        vert_int.astype(np.float64) / res
        - 0.5
        + vert_offsets.astype(np.float64) / (res * 2.0)
    )
    trimesh.Trimesh(vertices=vert_with_offset, faces=faces).export(
        os.path.join(out_dir, f"{base_name}_pred.obj")
    )
    trimesh.Trimesh(vertices=vert_int.astype(np.float64), faces=faces).export(
        os.path.join(out_dir, f"{base_name}_pred_coords.obj")
    )
