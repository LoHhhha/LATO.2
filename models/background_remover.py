import os

import torch
from PIL import Image
from torchvision import transforms

from utils import logging

REMBG_REPO_ID = "ZhengPeng7/BiRefNet"


class BackgroundRemover:
    """BiRefNet matting model (https://huggingface.co/ZhengPeng7/BiRefNet).

    The model lives in ``ckpt_dir/BiRefNet``: reused when present, downloaded
    there otherwise. It ships its own code, which is executed on load
    (``trust_remote_code``).
    """

    def __init__(self, ckpt_dir: str, device: torch.device):
        from transformers import AutoModelForImageSegmentation

        self.model_dir = os.path.join(ckpt_dir, REMBG_REPO_ID.split("/")[-1])
        if not all(
            os.path.isfile(os.path.join(self.model_dir, f))
            for f in ("config.json", "model.safetensors")
        ):
            from huggingface_hub import snapshot_download

            logging.info(f"{self.model_dir} not found; downloading {REMBG_REPO_ID}")
            snapshot_download(
                REMBG_REPO_ID,
                local_dir=self.model_dir,
                allow_patterns=["*.json", "*.py", "*.safetensors"],
            )
        self.model = (
            AutoModelForImageSegmentation.from_pretrained(
                self.model_dir, trust_remote_code=True
            )
            .to(device)
            .eval()
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> Image.Image:
        """RGB image -> RGBA image whose alpha is the predicted foreground matte."""
        param = next(self.model.parameters())
        x = self.transform(image).unsqueeze(0).to(device=param.device, dtype=param.dtype)
        matte = self.model(x)[-1].sigmoid()[0].float().cpu()  # (1, 1024, 1024)
        mask = transforms.ToPILImage()(matte).resize(image.size)
        out = image.copy()
        out.putalpha(mask)
        return out
