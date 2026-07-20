import torch.nn as nn


class OffsetHead(nn.Module):
    def __init__(self, feat_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, int(feat_dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(feat_dim * mlp_ratio), 3),
            nn.Tanh(),
        )

    def forward(self, vtx_feats):
        """
        Input:
            vtx_feats: [N, feat_dim]
        Output:
            offsets: [N, 3], in range (-1, 1)
        """
        offsets = self.mlp(vtx_feats)  # [N, 3], (-1, 1)
        return offsets
