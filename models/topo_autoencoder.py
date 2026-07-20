from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch import nn

from modules.pointnet import Pointnet
from modules.transformer.hybrid import (
    HybridGraphFlashStack,
    FlashVarlenTransformerBlock,
)
from modules.utils import manual_cast, str_to_dtype
from modules.transformer.blocks import (
    PointEmbed,
    RotaryPositionPhasesEmbedder,
    MaskedTransformerCrossAttnBlock,
)


class TopologyEncoderHybrid(nn.Module):
    def __init__(
        self,
        z_dim: int = 32,
        hidden_dim: int = 384,
        num_heads: int = 6,
        num_discrete: int = 256,
        dtype: str = "float32",
        num_hybrid_stages: int = 2,
        num_flash_per_stage: int = 1,
        use_gradient_checkpointing: bool = False,
        pc_cross_attn: bool = False,
    ):
        super().__init__()
        self.dtype = str_to_dtype(dtype)
        self.num_discrete = num_discrete
        self.pc_cross_attn = bool(pc_cross_attn)

        head_dim = hidden_dim // num_heads
        self.rope = RotaryPositionPhasesEmbedder(head_dim=head_dim, dim=3)

        self.backbone = HybridGraphFlashStack(
            hidden_size=hidden_dim,
            num_heads=num_heads,
            num_stages=num_hybrid_stages,
            num_flash_per_stage=num_flash_per_stage,
            gradient_checkpointing=use_gradient_checkpointing,
        )
        if self.pc_cross_attn:
            self.pc_cross_blocks = nn.ModuleList(
                [
                    MaskedTransformerCrossAttnBlock(
                        hidden_dim, num_heads, cond_dim=hidden_dim
                    )
                    for _ in range(num_hybrid_stages)
                ]
            )
        self.z_proj = nn.Linear(hidden_dim, z_dim * 2)
        nn.init.zeros_(self.z_proj.weight)
        nn.init.zeros_(self.z_proj.bias)

    def forward(
        self,
        verts: torch.Tensor,
        pc_tokens: Optional[torch.Tensor],
        point_embedder: PointEmbed,
        verts_mask: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None,
    ):
        rope_phases = self.rope(verts.long())
        coords = (verts + 0.5) / self.num_discrete * 2 - 1
        vert_tokens = point_embedder(coords)
        vert_tokens = manual_cast(vert_tokens, self.dtype)

        adj_mask = None
        if adj_matrix is not None:
            b, n, _ = adj_matrix.shape
            eye = torch.eye(n, device=adj_matrix.device, dtype=torch.bool).unsqueeze(0)
            adj_mask = adj_matrix.bool() | eye

        if self.pc_cross_attn:
            if pc_tokens is None:
                raise ValueError(
                    "pc_tokens required when encoder pc_cross_attn is enabled"
                )
            for stage, ca_block in zip(self.backbone.stages, self.pc_cross_blocks):
                vert_tokens = stage(
                    vert_tokens,
                    x_mask=verts_mask,
                    adj_matrix=adj_mask,
                    rope_phases=rope_phases,
                )
                vert_tokens = ca_block(
                    vert_tokens,
                    pc_tokens,
                    x_mask=verts_mask,
                    c_mask=None,
                )
        else:
            vert_tokens = self.backbone(
                vert_tokens,
                x_mask=verts_mask,
                adj_matrix=adj_mask,
                rope_phases=rope_phases,
            )
        z = self.z_proj(vert_tokens)
        z = manual_cast(z, self.dtype)
        return z


class TopologyDecoderHybrid(nn.Module):
    def __init__(
        self,
        z_dim: int = 32,
        hidden_dim: int = 384,
        num_heads: int = 6,
        num_discrete: int = 256,
        dtype: str = "float32",
        num_hybrid_stages: int = 2,
        num_flash_per_stage: int = 1,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.dtype = str_to_dtype(dtype)
        self.num_discrete = num_discrete
        self.input_proj = nn.Linear(z_dim, hidden_dim)
        self.backbone = HybridGraphFlashStack(
            hidden_size=hidden_dim,
            num_heads=num_heads,
            num_stages=num_hybrid_stages,
            num_flash_per_stage=num_flash_per_stage,
            gradient_checkpointing=use_gradient_checkpointing,
        )

    def forward(self, z: torch.Tensor, verts_mask: Optional[torch.Tensor] = None):
        h = self.input_proj(z)
        h = manual_cast(h, self.dtype)
        h = self.backbone(
            h,
            x_mask=verts_mask,
            adj_matrix=None,
            rope_phases=None,
        )
        return h


class TopologyConnectionPredictor(nn.Module):
    def __init__(self, hidden_dim: int = 384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, vert_feat_u: torch.Tensor, vert_feat_v: torch.Tensor):
        pair_feat_0 = torch.cat([vert_feat_u, vert_feat_v], dim=-1)
        pair_feat_1 = torch.cat([vert_feat_v, vert_feat_u], dim=-1)
        h = (self.mlp(pair_feat_0) + self.mlp(pair_feat_1)) / 2.0
        return h.squeeze(-1)


class TopologyVAE(nn.Module):
    def __init__(
        self,
        z_dim: int = 32,
        hidden_dim: int = 384,
        pc_dim: int = 15,
        inner_pc_dim: int = 256,
        num_heads: int = 6,
        num_discrete: int = 256,
        dtype: str = "float32",
        num_hybrid_stages: int = 2,
        num_flash_per_stage: int = 1,
        num_connection_blocks: Optional[int] = None,
        use_gradient_checkpointing: bool = False,
        encoder_pc_cross_attn: bool = False,
    ):
        super().__init__()
        self.dtype = str_to_dtype(dtype)
        self.num_discrete = num_discrete
        self.encoder_pc_cross_attn = bool(encoder_pc_cross_attn)

        self.point_embed = PointEmbed(hidden_dim=hidden_dim, dim=hidden_dim)
        self.point_net = Pointnet(
            in_channels=pc_dim,
            out_channels=inner_pc_dim,
            hidden_dim=256,
            n_blocks=5,
        )
        self.point_fusion = nn.Linear(hidden_dim + inner_pc_dim, hidden_dim)

        self.encoder = TopologyEncoderHybrid(
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_discrete=num_discrete,
            dtype=dtype,
            num_hybrid_stages=num_hybrid_stages,
            num_flash_per_stage=num_flash_per_stage,
            use_gradient_checkpointing=use_gradient_checkpointing,
            pc_cross_attn=self.encoder_pc_cross_attn,
        )
        self.decoder = TopologyDecoderHybrid(
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_discrete=num_discrete,
            dtype=dtype,
            num_hybrid_stages=num_hybrid_stages,
            num_flash_per_stage=num_flash_per_stage,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.connection_predictor = TopologyConnectionPredictor(hidden_dim=hidden_dim)

        head_dim = hidden_dim // num_heads
        self.connection_rope = RotaryPositionPhasesEmbedder(head_dim=head_dim, dim=3)

        n_conn = (
            num_connection_blocks
            if num_connection_blocks is not None
            else (num_hybrid_stages * (1 + num_flash_per_stage))
        )
        self.connection_transformer_blocks = nn.ModuleList(
            [
                FlashVarlenTransformerBlock(
                    hidden_dim,
                    num_heads,
                    gradient_checkpointing=use_gradient_checkpointing,
                )
                for _ in range(n_conn)
            ]
        )

    def encode(
        self,
        verts: torch.Tensor,
        pc_tokens: torch.Tensor,
        verts_mask: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None,
    ):
        moments = self.encoder(
            verts=verts,
            pc_tokens=pc_tokens,
            point_embedder=self.point_embed,
            verts_mask=verts_mask,
            adj_matrix=adj_matrix,
        )
        mean, logvar = moments.chunk(2, dim=-1)
        return mean, logvar

    def decode(
        self,
        z: torch.Tensor,
        verts: torch.Tensor,
        verts_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 20000,
        threshold: float = 0.0,
    ) -> List[np.ndarray]:
        # return: list of [N, 2] numpy arrays of predicted edges for each batch item
        verts_feat = self.decoder(z=z, verts_mask=verts_mask)

        rope = self.connection_rope(verts.long())
        for block in self.connection_transformer_blocks:
            verts_feat = block(verts_feat, verts_mask, rope_phases=rope)

        all_pred_edges_list = []
        for i in range(verts_mask.shape[0]):
            valid = verts_mask[i]
            valid_verts = verts[i][valid]
            valid_feats = verts_feat[i][valid]
            num_valid = int(valid_verts.shape[0])

            u_idx, v_idx = torch.triu_indices(
                num_valid, num_valid, offset=1, device=z.device
            )
            pred_edges_list = []
            for i in range(0, u_idx.numel(), chunk_size):
                cu = u_idx[i : i + chunk_size]
                cv = v_idx[i : i + chunk_size]
                logits = self.connection_predictor(
                    valid_feats[cu].unsqueeze(0),
                    valid_feats[cv].unsqueeze(0),
                ).squeeze(0)
                take = logits > threshold
                if bool(take.any()):
                    pred_edges_list.append(torch.stack([cu[take], cv[take]], dim=-1))

            pred_edges = (
                torch.cat(pred_edges_list, dim=0).cpu().numpy()
                if pred_edges_list
                else np.empty((0, 2), dtype=np.int64)
            )
            all_pred_edges_list.append(pred_edges)

        return all_pred_edges_list
