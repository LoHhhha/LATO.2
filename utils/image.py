import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_rgba(path: str, remove_background) -> Image.Image:
    """Open an image and matte it with ``remove_background`` (RGB -> RGBA).

    Any transparency the file already has is flattened onto white first, so every
    input takes the same path; the image is limited to 1024 px on its longer side
    as in TRELLIS' preprocess_image.
    """
    image = Image.open(path)
    if image.mode != "RGB":
        rgba = image.convert("RGBA")
        image = Image.new("RGB", rgba.size, (255, 255, 255))
        image.paste(rgba, (0, 0), rgba)
    scale = min(1.0, 1024 / max(image.size))
    if scale < 1:
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return remove_background(image)


def crop_foreground(
    rgba: Image.Image, padding: float = 1.2, size: int = 518
) -> np.ndarray:
    """Square-crop an RGBA image around its foreground and composite it on black.

    This is TRELLIS' preprocess_image and also the framing of the LATO.2
    conditioning renders (WhiteModelRenderer crop_to_object): the crop side is
    the larger side of the alpha > 0.8 bounding box times ``padding``, resized to
    ``size``. Returns a uint8 (size, size, 3) array.
    """
    alpha = np.array(rgba)[:, :, 3]
    ys, xs = np.nonzero(alpha > 0.8 * 255)
    if xs.size == 0:
        raise ValueError("no foreground found (alpha > 0.8)")
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(max(x1 - x0, y1 - y0) * padding)
    if side < 2:
        raise ValueError("foreground is too small to crop")
    box = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)
    out = rgba.crop(box).resize((size, size), Image.Resampling.LANCZOS)
    out = np.array(out).astype(np.float32) / 255
    out = out[:, :, :3] * out[:, :, 3:4]
    return (out * 255).astype(np.uint8)
