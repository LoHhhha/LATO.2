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
