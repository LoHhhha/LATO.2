from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modules.transformer.blocks import RotaryPositionPhasesEmbedder, TimestepEmbedder
from modules.attention import (
    can_flash_varlen,
    flash_varlen_self_attention,
    flash_varlen_cross_attention,
    sdpa_padding_mask,
)
from modules.norm import RMSNorm
from modules.utils import modulate


class TopologySiTBlockFlashVarlen(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        qk_norm_eps: float = 1e-5,
        qk_norm_variance_in_fp32: bool = True,
        with_cross_attn: bool = False,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.with_cross_attn = bool(with_cross_attn)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=True)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size, bias=True),
        )
        self._n_adaln_chunks = 7 if self.with_cross_attn else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, self._n_adaln_chunks * hidden_size, bias=True),
        )
        self.dropout = dropout
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.norm_q, self.norm_k = (
            RMSNorm(
                self.head_dim,
                eps=qk_norm_eps,
                elementwise_affine=True,
                variance_in_fp32=qk_norm_variance_in_fp32,
            ),
            RMSNorm(
                self.head_dim,
                eps=qk_norm_eps,
                elementwise_affine=True,
                variance_in_fp32=qk_norm_variance_in_fp32,
            ),
        )

        if self.with_cross_attn:
            self.norm_ca = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.q_ca = nn.Linear(hidden_size, hidden_size, bias=True)
            self.kv_ca = nn.Linear(hidden_size, hidden_size * 2, bias=True)
            self.proj_ca_out = nn.Linear(hidden_size, hidden_size, bias=True)
            self.norm_q_ca, self.norm_k_ca = (
                RMSNorm(
                    self.head_dim,
                    eps=qk_norm_eps,
                    elementwise_affine=True,
                    variance_in_fp32=qk_norm_variance_in_fp32,
                ),
                RMSNorm(
                    self.head_dim,
                    eps=qk_norm_eps,
                    elementwise_affine=True,
                    variance_in_fp32=qk_norm_variance_in_fp32,
                ),
            )

    def _forward_once(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        rope_phases: torch.Tensor,
        cond_emb: torch.Tensor | None,
        cond_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        chunks = self.adaLN_modulation(c).chunk(self._n_adaln_chunks, dim=1)
        if self.with_cross_attn:
            shift_msa, scale_msa, gate_msa, gate_mca, shift_mlp, scale_mlp, gate_mlp = (
                chunks
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        h = modulate(self.norm1(x), shift_msa, scale_msa)
        b, n, d = h.shape
        qkv = (
            self.qkv(h)
            .view(b, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)
        q = RotaryPositionPhasesEmbedder.apply_rotary_embedding(q, rope_phases)
        k = RotaryPositionPhasesEmbedder.apply_rotary_embedding(k, rope_phases)

        x_mask = None if key_padding_mask is None else ~key_padding_mask.bool()
        if can_flash_varlen(q, x_mask):
            attn_out = flash_varlen_self_attention(q, k, v, x_mask)
        elif x_mask is not None:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=sdpa_padding_mask(x_mask),
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
            )

        attn_out = attn_out.transpose(1, 2).reshape(b, n, d)
        x = x + gate_msa.unsqueeze(1) * self.proj_out(attn_out)

        if self.with_cross_attn and cond_emb is not None:
            h_ca = self.norm_ca(x)
            nk = cond_emb.shape[1]
            q_ca = (
                self.q_ca(h_ca)
                .view(b, n, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            kv_ca = (
                self.kv_ca(cond_emb)
                .view(b, nk, 2, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            k_ca, v_ca = kv_ca[0], kv_ca[1]
            if self.norm_q_ca is not None:
                q_ca = self.norm_q_ca(q_ca)
            if self.norm_k_ca is not None:
                k_ca = self.norm_k_ca(k_ca)

            q_mask_bool = (
                torch.ones(b, n, dtype=torch.bool, device=q_ca.device)
                if key_padding_mask is None
                else ~key_padding_mask.bool()
            )
            k_mask_bool = (
                torch.ones(b, nk, dtype=torch.bool, device=q_ca.device)
                if cond_mask is None
                else cond_mask.bool()
            )

            if can_flash_varlen(q_ca, q_mask_bool):
                ca_out = flash_varlen_cross_attention(
                    q_ca, k_ca, v_ca, q_mask_bool, k_mask_bool
                )
            else:
                k_attn_mask = k_mask_bool.view(b, 1, 1, nk)
                ca_out = F.scaled_dot_product_attention(
                    q_ca, k_ca, v_ca, attn_mask=k_attn_mask, dropout_p=0.0
                )
            ca_out = ca_out.transpose(1, 2).reshape(b, n, d)
            x = x + gate_mca.unsqueeze(1) * self.proj_ca_out(ca_out)

        h2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h2)
        if key_padding_mask is not None:
            valid = ~key_padding_mask.bool()
            x = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        rope_phases: torch.Tensor,
        cond_emb: torch.Tensor | None = None,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(
                self._forward_once,
                x,
                c,
                key_padding_mask,
                rope_phases,
                cond_emb,
                cond_mask,
                use_reentrant=False,
            )
        return self._forward_once(
            x, c, key_padding_mask, rope_phases, cond_emb, cond_mask
        )


class TopologyFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


def _get_1d_sincos_embed(n: int, dim: int) -> torch.Tensor:
    assert dim % 2 == 0
    pos = torch.arange(n, dtype=torch.float32)
    omega = torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
    omega = 1.0 / (10000**omega)
    out = pos[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)


class TopologySiTFlow(nn.Module):
    def __init__(
        self,
        z_dim: int,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        max_vertices: int = 8192,
        num_discrete: int = 1024,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        cond_in_dim: int = 0,
        cond_dropout_prob: float = 0.0,
        qk_norm_eps: float = 1e-5,
        qk_norm_variance_in_fp32: bool = True,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.max_vertices = max_vertices
        self.num_discrete = int(num_discrete)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.cond_in_dim = int(cond_in_dim)
        self.cond_dropout_prob = float(cond_dropout_prob)
        self.qk_norm_eps = float(qk_norm_eps)
        self.qk_norm_variance_in_fp32 = bool(qk_norm_variance_in_fp32)

        self.input_proj = nn.Linear(z_dim, hidden_size, bias=True)
        self.coord_embed = nn.Sequential(
            nn.Linear(3, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.rope = RotaryPositionPhasesEmbedder(
            head_dim=hidden_size // num_heads, dim=3
        )

        if self.cond_in_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(self.cond_in_dim, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
            self.null_token = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.cond_proj = None
            self.null_token = None

        pe = _get_1d_sincos_embed(max_vertices, hidden_size)
        self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)

        with_cross_attn = self.cond_in_dim > 0
        self.blocks = nn.ModuleList(
            [
                TopologySiTBlockFlashVarlen(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    gradient_checkpointing=self.gradient_checkpointing,
                    with_cross_attn=with_cross_attn,
                    qk_norm_eps=self.qk_norm_eps,
                    qk_norm_variance_in_fp32=self.qk_norm_variance_in_fp32,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = TopologyFinalLayer(hidden_size, z_dim)

    def _prepare_cond(
        self,
        b: int,
        cond: torch.Tensor | None,
        cond_mask: torch.Tensor | None,
        cond_drop_override: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.cond_proj is None:
            return None, None

        if cond is None:
            null_emb = self.null_token.view(1, 1, -1).expand(b, 1, -1).contiguous()
            mask_out = torch.ones(b, 1, dtype=torch.bool, device=device)
            return null_emb, mask_out

        if cond.dim() != 3 or cond.shape[0] != b or cond.shape[-1] != self.cond_in_dim:
            raise ValueError(
                f"cond shape {tuple(cond.shape)} expected ({b}, K, {self.cond_in_dim})"
            )
        k = cond.shape[1]
        cond_emb = self.cond_proj(cond)
        null = self.null_token.view(1, 1, -1).to(dtype=cond_emb.dtype)

        if cond_mask is None:
            mask_out = torch.ones(b, k, dtype=torch.bool, device=device)
        else:
            mask_out = cond_mask.to(device=device, dtype=torch.bool)
            if mask_out.shape != (b, k):
                raise ValueError(
                    f"cond_mask shape {tuple(mask_out.shape)} expected ({b}, {k})"
                )

        drop: torch.Tensor | None = None
        if cond_drop_override is not None:
            drop = cond_drop_override.to(device=device, dtype=torch.bool).reshape(b)
        elif self.training and self.cond_dropout_prob > 0.0:
            drop = torch.rand(b, device=device) < self.cond_dropout_prob

        if drop is not None:
            null_emb = null.expand(b, k, -1)
            cond_emb = torch.where(drop.view(b, 1, 1), null_emb, cond_emb)
            mask_out = torch.where(drop.view(b, 1), torch.ones_like(mask_out), mask_out)
        return cond_emb, mask_out

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        verts: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor | None = None,
        cond_mask: torch.Tensor | None = None,
        cond_drop_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        if n > self.max_vertices:
            raise ValueError(f"Sequence length {n} > max_vertices {self.max_vertices}")
        if self.cond_in_dim == 0:
            if cond is not None:
                raise ValueError("TopologySiT(cond_in_dim=0): pass cond=None")
            if cond_mask is not None:
                raise ValueError("TopologySiT(cond_in_dim=0): cond_mask is unused")
            if cond_drop_override is not None:
                raise ValueError(
                    "TopologySiT(cond_in_dim=0): cond_drop_override is unused"
                )

        coords = ((verts.float() + 0.5) / self.num_discrete) * 2.0 - 1.0
        h = self.input_proj(x) + self.pos_embed[:, :n, :] + self.coord_embed(coords)
        c = self.t_embedder(t)

        cond_emb, cond_mask_eff = self._prepare_cond(
            b, cond, cond_mask, cond_drop_override, x.device
        )

        key_padding_mask = ~mask
        rope_phases = self.rope(verts.long())
        for block in self.blocks:
            h = block(h, c, key_padding_mask, rope_phases, cond_emb, cond_mask_eff)
        out = self.final_layer(h, c)
        out = torch.where(mask.unsqueeze(-1), out, torch.zeros_like(out))
        return out
