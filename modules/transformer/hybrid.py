from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..attention import (
    can_flash_varlen,
    flash_varlen_self_attention,
    graph_adj_varlen_attention,
    sdpa_padding_mask,
)
from .blocks import RotaryPositionPhasesEmbedder


class GraphAttnVarlenBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size, bias=True),
        )
        self.scale_msa = nn.Parameter(torch.zeros(hidden_size))
        self.scale_mlp = nn.Parameter(torch.zeros(hidden_size))
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def _forward_once(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor],
        adj_matrix: Optional[torch.Tensor],
        rope_phases: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv = (
            self.qkv(self.norm1(x))
            .view(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rope_phases is not None:
            q = RotaryPositionPhasesEmbedder.apply_rotary_embedding(q, rope_phases)
            k = RotaryPositionPhasesEmbedder.apply_rotary_embedding(k, rope_phases)

        if x_mask is None:
            attn_mask = None
            if adj_matrix is not None:
                adj_mask = adj_matrix.bool()
                eye = torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
                attn_mask = (adj_mask | eye).unsqueeze(1)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            attn_out = graph_adj_varlen_attention(q, k, v, x_mask, adj_matrix)

        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        x = x + self.scale_msa * self.proj_out(attn_out)
        x = x + self.scale_mlp * self.ffn(self.norm2(x))
        if x_mask is not None:
            x = torch.where(x_mask.unsqueeze(-1), x, torch.zeros_like(x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor],
        adj_matrix: Optional[torch.Tensor],
        rope_phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(
                self._forward_once,
                x,
                x_mask,
                adj_matrix,
                rope_phases,
                use_reentrant=False,
            )
        return self._forward_once(x, x_mask, adj_matrix, rope_phases)


class FlashVarlenTransformerBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size, bias=True),
        )
        self.scale_msa = nn.Parameter(torch.zeros(hidden_size))
        self.scale_mlp = nn.Parameter(torch.zeros(hidden_size))
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def _forward_once(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor],
        rope_phases: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv = (
            self.qkv(self.norm1(x))
            .view(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rope_phases is not None:
            q = RotaryPositionPhasesEmbedder.apply_rotary_embedding(q, rope_phases)
            k = RotaryPositionPhasesEmbedder.apply_rotary_embedding(k, rope_phases)

        if can_flash_varlen(q, x_mask):
            attn_out = flash_varlen_self_attention(q, k, v, x_mask)
        elif x_mask is not None:
            pad_mask = sdpa_padding_mask(x_mask)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=pad_mask)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None)

        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        x = x + self.scale_msa * self.proj_out(attn_out)
        x = x + self.scale_mlp * self.ffn(self.norm2(x))
        if x_mask is not None:
            x = torch.where(x_mask.unsqueeze(-1), x, torch.zeros_like(x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor],
        rope_phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(
                self._forward_once,
                x,
                x_mask,
                rope_phases,
                use_reentrant=False,
            )
        return self._forward_once(x, x_mask, rope_phases)


class HybridGraphFlashStage(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_flash: int,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.graph_block = GraphAttnVarlenBlock(
            hidden_size, num_heads, gradient_checkpointing=gradient_checkpointing
        )
        self.flash_blocks = nn.ModuleList(
            [
                FlashVarlenTransformerBlock(
                    hidden_size,
                    num_heads,
                    gradient_checkpointing=gradient_checkpointing,
                )
                for _ in range(num_flash)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor],
        adj_matrix: Optional[torch.Tensor],
        rope_phases: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.graph_block(
            x, x_mask=x_mask, adj_matrix=adj_matrix, rope_phases=rope_phases
        )
        for fb in self.flash_blocks:
            x = fb(x, x_mask=x_mask, rope_phases=rope_phases)
        return x


class HybridGraphFlashStack(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_stages: int,
        num_flash_per_stage: int,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                HybridGraphFlashStage(
                    hidden_size,
                    num_heads,
                    num_flash_per_stage,
                    gradient_checkpointing=gradient_checkpointing,
                )
                for _ in range(num_stages)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor],
        adj_matrix: Optional[torch.Tensor],
        rope_phases: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x, x_mask=x_mask, adj_matrix=adj_matrix, rope_phases=rope_phases)
        return x
