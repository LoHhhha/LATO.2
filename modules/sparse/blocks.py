from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import zero_module
from ..norm import LayerNorm32
from .. import sparse as sp


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        self.use_checkpoint = use_checkpoint

        assert not (
            downsample and upsample
        ), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(
            sp.SparseConv3d(self.out_channels, self.out_channels, 3)
        )

        self.skip_connection = (
            sp.SparseLinear(channels, self.out_channels)
            if channels != self.out_channels
            else nn.Identity()
        )
        self.updown = None
        if self.downsample:
            self.updown = sp.SparseDownsample(2)
        elif self.upsample:
            self.updown = sp.SparseUpsample(2)

    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.updown is not None:
            x = self.updown(x)
        return x

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        x = self._updown(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)

        return h

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False
            )
        else:
            return self._forward(x)
