# MIT License

# Copyright (c) Microsoft Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# Copied and trimmed (inference only) from
# https://github.com/microsoft/TRELLIS/blob/main/trellis/models/sparse_structure_flow.py

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.transformer import AbsolutePositionEmbedder, TimestepEmbedder
from modules.utils import convert_module_to_f16

from .modules import ModulatedTransformerCrossBlock


class SparseStructureFlowModel(nn.Module):
    """TRELLIS stage-1 flow transformer over the dense sparse-structure latent grid."""

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        if patch_size != 1 or pe_mode != "ape":
            raise NotImplementedError(
                f"only patch_size=1 / pe_mode='ape' sparse-structure flows are "
                f"supported, got patch_size={patch_size}, pe_mode={pe_mode!r}"
            )
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.share_mod = share_mod
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        coords = torch.meshgrid(
            *[torch.arange(resolution) for _ in range(3)], indexing="ij"
        )
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)
        self.register_buffer(
            "pos_emb", AbsolutePositionEmbedder(model_channels, 3)(coords)
        )

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_layer = nn.Linear(model_channels, out_channels)

        if use_fp16:
            self.blocks.apply(convert_module_to_f16)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """x: (B, C, R, R, R) noisy latent, t: (B,) in [0, 1000], cond: (B, L, C_cond)."""
        res = self.resolution
        assert [*x.shape] == [x.shape[0], self.in_channels, res, res, res], (
            f"Input shape mismatch, got {x.shape}, expected "
            f"{[x.shape[0], self.in_channels, res, res, res]}"
        )

        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        return h.permute(0, 2, 1).view(h.shape[0], h.shape[2], res, res, res).contiguous()
