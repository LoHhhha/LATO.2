# MIT License

# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 VAST-AI-Research and contributors.

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

from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000**self.freqs)

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        N, D = x.shape
        assert (
            D == self.in_channels
        ), "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat(
                [
                    embed,
                    torch.zeros(N, self.channels - embed.shape[1], device=embed.device),
                ],
                dim=-1,
            )
        return embed


class RotaryPositionPhasesEmbedder(nn.Module):
    def __init__(
        self,
        head_dim: int,
        dim: int = 3,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
    ):
        super().__init__()
        assert head_dim % 2 == 0, "Head dim must be divisible by 2"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = rope_freq[0] / (rope_freq[1] ** (self.freqs))

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    @staticmethod
    def apply_rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        if phases.ndim == 3:
            phases = phases.unsqueeze(1)
        x_rotated = x_complex * phases
        x_embed = (
            torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        )
        return x_embed

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        assert indices.shape[-1] == self.dim, f"Last dim of indices must be {self.dim}"
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[-1] < self.head_dim // 2:
            padn = self.head_dim // 2 - phases.shape[-1]
            phases = torch.cat(
                [
                    phases,
                    torch.polar(
                        torch.ones(*phases.shape[:-1], padn, device=phases.device),
                        torch.zeros(*phases.shape[:-1], padn, device=phases.device),
                    ),
                ],
                dim=-1,
            )
        return phases


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat(
                    [
                        e,
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        e,
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                        e,
                    ]
                ),
            ]
        )
        self.register_buffer("basis", e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        dt = self.mlp.weight.dtype
        if input.dtype != dt:
            input = input.to(dtype=dt)
        basis = self.basis.to(dtype=dt)
        embed = self.mlp(torch.cat([self.embed(input, basis), input], dim=2))
        return embed


class MaskedTransformerCrossAttnBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, cond_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.q_cross = nn.Linear(hidden_size, hidden_size, bias=True)
        self.kv_cross = nn.Linear(cond_dim, hidden_size * 2, bias=True)
        self.proj_out_cross = nn.Linear(hidden_size, hidden_size, bias=True)
        self.scale_cross = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        c_tokens: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        c_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, d = x.shape
        q_c = (
            self.q_cross(self.norm(x))
            .view(b, n, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        kv_c = (
            self.kv_cross(c_tokens)
            .view(b, c_tokens.shape[1], 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k_c, v_c = kv_c[0], kv_c[1]
        cross_attn_mask = c_mask.view(b, 1, 1, -1) if c_mask is not None else None
        cross_out = F.scaled_dot_product_attention(
            q_c,
            k_c,
            v_c,
            attn_mask=cross_attn_mask,
        )
        cross_out = cross_out.transpose(1, 2).reshape(b, n, d)
        x = x + self.scale_cross * self.proj_out_cross(cross_out)
        if x_mask is not None:
            x = torch.where(x_mask.unsqueeze(-1), x, torch.zeros_like(x))
        return x
