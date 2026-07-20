from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import open3d as o3d
except Exception as _e:
    o3d = None
    _OPEN3D_IMPORT_ERROR = _e


ColorLike = Union[Sequence[float], np.ndarray]


def _to_rgb01(color: ColorLike) -> Tuple[float, float, float]:
    c = np.asarray(color, dtype=np.float64).reshape(-1)[:3]
    if c.max() > 1.0 + 1e-6:
        c = c / 255.0
    return float(c[0]), float(c[1]), float(c[2])


def _axis_index(up_axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[up_axis.lower()]


def _orbit_eye(
    center: np.ndarray,
    distance: float,
    azimuth_deg: float,
    elevation_deg: float,
    up_axis: str,
) -> Tuple[np.ndarray, np.ndarray]:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    ce = np.cos(el)

    horiz = distance * ce
    vert = distance * np.sin(el)
    ai = _axis_index(up_axis)
    offset = np.zeros(3, dtype=np.float64)

    plane_axes = [i for i in range(3) if i != ai]
    offset[plane_axes[0]] = horiz * np.cos(az)
    offset[plane_axes[1]] = horiz * np.sin(az)
    offset[ai] = vert
    eye = center + offset
    up = np.zeros(3, dtype=np.float64)
    up[ai] = 1.0
    return eye, up


class WhiteModelRenderer:
    def __init__(
        self,
        img_res: int = 512,
        mesh_color: ColorLike = (0.78, 0.78, 0.82),
        bg_color: ColorLike = (1.0, 1.0, 1.0),
        up_axis: str = "y",
        add_ground: bool = True,
        shadow: bool = True,
        elevation_range: Tuple[float, float] = (15.0, 40.0),
        azimuth_range: Tuple[float, float] = (0.0, 360.0),
        camera_distance: float = 1.8,
        fov: float = 50.0,
        ground_color: ColorLike = (0.92, 0.92, 0.92),
        sun_intensity: float = 90000.0,
        ambient_intensity: float = 32000.0,
        crop_to_object: bool = False,
        crop_padding: float = 1.2,
    ):
        if o3d is None:
            raise ImportError(
                f"open3d is required for WhiteModelRenderer but failed to import: {_OPEN3D_IMPORT_ERROR}"
            )
        self.img_res = int(img_res)
        self.mesh_color = _to_rgb01(mesh_color)
        self.bg_color = _to_rgb01(bg_color)
        self.up_axis = up_axis.lower()
        self.add_ground = add_ground
        self.shadow = shadow
        self.elevation_range = elevation_range
        self.azimuth_range = azimuth_range
        self.camera_distance = float(camera_distance)
        self.fov = float(fov)
        self.ground_color = _to_rgb01(ground_color)
        self.sun_intensity = float(sun_intensity)
        self.ambient_intensity = float(ambient_intensity)
        self.crop_to_object = crop_to_object
        self.crop_padding = float(crop_padding)

        self._renderer = None
        self._rng = np.random.default_rng()

    def _ensure_renderer(self):
        if self._renderer is None:
            self._renderer = o3d.visualization.rendering.OffscreenRenderer(
                self.img_res, self.img_res
            )
        return self._renderer

    def _make_o3d_mesh(self, vertices: np.ndarray, faces: np.ndarray):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            np.asarray(vertices, dtype=np.float64)
        )
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
        mesh.compute_vertex_normals()
        return mesh

    def _make_ground(self, mesh_min: np.ndarray, mesh_max: np.ndarray):
        ai = _axis_index(self.up_axis)
        center = (mesh_min + mesh_max) / 2.0
        extent = float(np.max(mesh_max - mesh_min))
        size = max(extent * 6.0, 4.0)

        plane_axes = [i for i in range(3) if i != ai]
        bottom = mesh_min[ai] - extent * 0.02

        corners_2d = (
            np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * size
        )
        verts = np.zeros((4, 3), dtype=np.float64)
        for k, (a, b) in enumerate(corners_2d):
            verts[k, plane_axes[0]] = center[plane_axes[0]] + a
            verts[k, plane_axes[1]] = center[plane_axes[1]] + b
            verts[k, ai] = bottom
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        ground = o3d.geometry.TriangleMesh()
        ground.vertices = o3d.utility.Vector3dVector(verts)
        ground.triangles = o3d.utility.Vector3iVector(tris)
        ground.compute_vertex_normals()
        return ground

    def _lit_material(self, rgb: Tuple[float, float, float], roughness: float = 0.85):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [rgb[0], rgb[1], rgb[2], 1.0]
        mat.base_roughness = roughness
        mat.base_metallic = 0.0
        mat.base_reflectance = 0.4
        return mat

    def _setup_scene(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        mesh_color: Tuple[float, float, float],
    ):
        renderer = self._ensure_renderer()
        scene = renderer.scene
        scene.clear_geometry()
        scene.set_background(
            [self.bg_color[0], self.bg_color[1], self.bg_color[2], 1.0]
        )

        mesh = self._make_o3d_mesh(vertices, faces)
        scene.add_geometry("mesh", mesh, self._lit_material(mesh_color))

        mesh_min = np.asarray(vertices, dtype=np.float64).min(axis=0)
        mesh_max = np.asarray(vertices, dtype=np.float64).max(axis=0)
        if self.add_ground:
            ground = self._make_ground(mesh_min, mesh_max)
            scene.add_geometry(
                "ground", ground, self._lit_material(self.ground_color, roughness=0.95)
            )

        ai = _axis_index(self.up_axis)
        sun_dir = np.array([0.35, 0.35, 0.35])
        sun_dir[ai] = -1.0
        sun_dir = sun_dir / np.linalg.norm(sun_dir)

        scene.scene.set_sun_light(sun_dir.tolist(), [1.0, 1.0, 1.0], self.sun_intensity)
        scene.scene.enable_sun_light(True)
        scene.scene.set_indirect_light_intensity(self.ambient_intensity)

        center = (mesh_min + mesh_max) / 2.0
        return center

    def _object_mask_from_depth(self):
        renderer = self._renderer
        depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
        mask = np.isfinite(depth) & (depth > 0)
        return mask

    def _crop_resize_to_object(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from PIL import Image as _Image

        ys, xs = np.where(mask)
        if xs.size == 0:
            out = _Image.fromarray(rgb).resize(
                (self.img_res, self.img_res), _Image.LANCZOS
            )
            return np.asarray(out)

        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        size = int(max(x1 - x0, y1 - y0) * self.crop_padding)
        size = max(size, 1)
        half = size // 2
        bx0, by0, bx1, by1 = (
            int(round(cx - half)),
            int(round(cy - half)),
            int(round(cx - half)) + size,
            int(round(cy - half)) + size,
        )

        H, W = rgb.shape[:2]
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        sx0, sy0 = max(0, bx0), max(0, by0)
        sx1, sy1 = min(W, bx1), min(H, by1)
        if sx1 > sx0 and sy1 > sy0:
            canvas[sy0 - by0 : sy1 - by0, sx0 - bx0 : sx1 - bx0] = rgb[sy0:sy1, sx0:sx1]

        out = _Image.fromarray(canvas).resize(
            (self.img_res, self.img_res), _Image.LANCZOS
        )
        return np.ascontiguousarray(np.asarray(out))

    def render(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        num_views: int = 1,
        mesh_color: Optional[ColorLike] = None,
        azimuths: Optional[Sequence[float]] = None,
        elevations: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[dict]]:
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        rgb = self.mesh_color if mesh_color is None else _to_rgb01(mesh_color)

        center = self._setup_scene(vertices, faces, rgb)
        renderer = self._renderer

        images: List[np.ndarray] = []
        params: List[dict] = []
        for v in range(num_views):
            if azimuths is not None:
                az = float(azimuths[v])
            else:
                az = float(rng.uniform(*self.azimuth_range))
            if elevations is not None:
                el = float(elevations[v])
            else:
                el = float(rng.uniform(*self.elevation_range))

            eye, up = _orbit_eye(center, self.camera_distance, az, el, self.up_axis)
            renderer.setup_camera(self.fov, center.tolist(), eye.tolist(), up.tolist())

            img = renderer.render_to_image()
            arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            arr = arr.astype(np.uint8)

            if self.crop_to_object:
                mask = self._object_mask_from_depth()
                arr = self._crop_resize_to_object(arr, mask)

            images.append(np.ascontiguousarray(arr))
            params.append(
                {"azimuth": az, "elevation": el, "distance": self.camera_distance}
            )

        return images, params
