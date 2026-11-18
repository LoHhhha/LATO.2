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

# Copied and trimmed (decoder only) from
# https://github.com/microsoft/TRELLIS/blob/main/trellis/models/sparse_structure_vae.py

from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import convert_module_to_f16, zero_module

from .modules import ChannelLayerNorm32, GroupNorm32, pixel_shuffle_3d


def norm_layer(norm_type: str, *args, **kwargs) -> nn.Module:
    if norm_type == "group":
        return GroupNorm32(32, *args, **kwargs)
    elif norm_type == "layer":
        return ChannelLayerNorm32(*args, **kwargs)
    raise ValueError(f"Invalid norm type {norm_type}")


class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = norm_layer(norm_type, channels)
        self.norm2 = norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(
            nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
        )
        self.skip_connection = (
            nn.Conv3d(channels, self.out_channels, 1)
            if channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip_connection(x)


class UpsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_shuffle_3d(self.conv(x), 2)


class SparseStructureDecoder(nn.Module):
    """TRELLIS sparse-structure decoder: latent grid -> occupancy logits."""

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)

        # as upstream: norm_type only selects the output norm, the residual
        # blocks always use their default layer norm
        self.middle_block = nn.Sequential(
            *[ResBlock3d(channels[0], channels[0]) for _ in range(num_res_blocks_middle)]
        )

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([ResBlock3d(ch, ch) for _ in range(num_res_blocks)])
            if i < len(channels) - 1:
                self.blocks.append(UpsampleBlock3d(ch, channels[i + 1]))

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1),
        )

        if use_fp16:
            self.blocks.apply(convert_module_to_f16)
            self.middle_block.apply(convert_module_to_f16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = h.type(self.dtype)
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)
        h = h.type(x.dtype)
        return self.out_layer(h)
