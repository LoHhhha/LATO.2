import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from modules.pointnet import LocalPoolPointnet


class VDFEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        out_channels,
        scatter_type,
        n_blocks,
        resolution=64,
        use_checkpoint=False,
    ):
        super().__init__()
        self.pointnet = LocalPoolPointnet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            scatter_type=scatter_type,
        )

        self.resolution = resolution
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        p,
        sparse_coords,
        res=None,
        bbox_size=(-0.5, 0.5),
    ):
        """
        Input:
            p: [N, in_channels]
            sparse_coords: [M, 4], (b, z, y, x)
        Output:
            geo_feats: [N, out_channels]
        """
        if res is None:
            res = self.resolution

        if self.use_checkpoint and self.training:
            geo_feats = checkpoint(
                self.pointnet, p, sparse_coords, res, bbox_size, use_reentrant=False
            )
        else:
            geo_feats = self.pointnet(p, sparse_coords, res=res, bbox_size=bbox_size)

        return geo_feats
