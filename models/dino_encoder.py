import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DINO_GITHUB_REPO = "facebookresearch/dinov2"
DINO_LOCAL_REPO_DIRNAME = "facebookresearch_dinov2_main"


class DinoV2Encoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        hub_dir: str,
        img_res: int,
    ):
        super().__init__()
        self.img_res = int(img_res)

        hub_dir = os.path.abspath(os.path.expanduser(hub_dir))
        os.makedirs(hub_dir, exist_ok=True)
        torch.hub.set_dir(hub_dir)
        local_repo = os.path.join(hub_dir, DINO_LOCAL_REPO_DIRNAME)
        if os.path.isdir(local_repo):
            self.backbone = torch.hub.load(
                local_repo, model_name, source="local", pretrained=True
            )
        else:
            self.backbone = torch.hub.load(
                DINO_GITHUB_REPO, model_name, source="github", pretrained=True
            )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, images: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Images -> layer-normed DINO-v2 patch tokens (B, L, C).

        Accepts uint8 channels-last images — (H, W, 3) or (B, H, W, 3) in
        [0, 255], the dataset render format — or float channels-first
        (B, 3, H, W) in [0, 1]. Any resolution; resized to ``img_res``.
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(np.ascontiguousarray(images))
        if images.dim() == 3:
            images = images[None]
        if images.dtype == torch.uint8:
            images = images.permute(0, 3, 1, 2).float() / 255.0

        x = images.to(device=self.img_mean.device, dtype=torch.float32)
        if x.shape[-2:] != (self.img_res, self.img_res):
            x = F.interpolate(
                x, (self.img_res, self.img_res), mode="bicubic", align_corners=False
            )
        x = (x - self.img_mean) / self.img_std

        feats = self.backbone(x, is_training=True)["x_prenorm"]
        return F.layer_norm(feats, feats.shape[-1:])
