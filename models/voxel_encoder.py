from __future__ import annotations

import torch
import torch.nn as nn


def _safe_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    g = min(max_groups, num_channels)
    while g > 1 and num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


def _sincos_1d(n: int, dim: int) -> torch.Tensor:
    assert dim % 2 == 0 and dim > 0, f"sincos dim must be positive even, got {dim}"
    pos = torch.arange(n, dtype=torch.float32)
    omega = torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
    omega = 1.0 / (10000**omega)
    out = pos[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)


def _get_3d_sincos_embed(n: int, dim: int) -> torch.Tensor:
    axis_dim = (dim // 3) // 2 * 2  # split across 3 axes, round to even
    if axis_dim <= 0:
        raise ValueError(
            f"cond_in_dim={dim} too small for 3D sincos PE (need >= 6 so each axis gets a positive even slice)"
        )
    e = _sincos_1d(n, axis_dim)  # (n, axis_dim)ß
    pe_d = e[:, None, None, :].expand(n, n, n, axis_dim)
    pe_h = e[None, :, None, :].expand(n, n, n, axis_dim)
    pe_w = e[None, None, :, :].expand(n, n, n, axis_dim)
    pe = torch.cat([pe_d, pe_h, pe_w], dim=-1).reshape(n * n * n, 3 * axis_dim)
    if pe.shape[-1] < dim:
        pad = torch.zeros(pe.shape[0], dim - pe.shape[-1])
        pe = torch.cat([pe, pad], dim=-1)
    return pe


class _ResBlock3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _safe_group_norm(channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            _safe_group_norm(channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _DownBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, blocks_per_level: int) -> None:
        super().__init__()
        res_blocks: list[nn.Module] = [
            _ResBlock3d(in_ch) for _ in range(blocks_per_level)
        ]
        res_blocks.append(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        )
        res_blocks.append(_safe_group_norm(out_ch))
        res_blocks.append(nn.SiLU())
        self.net = nn.Sequential(*res_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VoxelFieldConditioner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_in_dim: int,
        *,
        num_downsamples: int = 2,
        base_channels: int = 32,
        channel_mult: int = 2,
        blocks_per_level: int = 2,
        pos_embed: str = "sincos",
    ) -> None:
        super().__init__()
        if num_downsamples < 0:
            raise ValueError(f"num_downsamples must be >= 0, got {num_downsamples}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if cond_in_dim <= 0:
            raise ValueError(f"cond_in_dim must be > 0, got {cond_in_dim}")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be > 0, got {base_channels}")
        if channel_mult < 1:
            raise ValueError(f"channel_mult must be >= 1, got {channel_mult}")
        if blocks_per_level < 0:
            raise ValueError(f"blocks_per_level must be >= 0, got {blocks_per_level}")
        pos_embed = str(pos_embed).lower()
        if pos_embed not in ("sincos", "none"):
            raise ValueError(
                f"pos_embed={pos_embed!r} unsupported (use 'sincos' or 'none')"
            )

        self.in_channels = int(in_channels)
        self.cond_in_dim = int(cond_in_dim)
        self.num_downsamples = int(num_downsamples)
        self.base_channels = int(base_channels)
        self.channel_mult = int(channel_mult)
        self.blocks_per_level = int(blocks_per_level)
        self.pos_embed = pos_embed

        self.stem = nn.Sequential(
            nn.Conv3d(
                self.in_channels,
                self.base_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.base_channels),
            nn.SiLU(),
        )

        down_blocks: list[nn.Module] = []
        ch = self.base_channels
        for _ in range(self.num_downsamples):
            out_ch = ch * self.channel_mult
            down_blocks.append(_DownBlock3d(ch, out_ch, self.blocks_per_level))
            ch = out_ch
        self.down_blocks = nn.ModuleList(down_blocks)
        self._final_channels = ch  # base_channels * channel_mult ** num_downsamples

        self.tail_blocks = nn.Sequential(
            *[_ResBlock3d(ch) for _ in range(blocks_per_level)]
        )

        self.proj = nn.Conv3d(
            self._final_channels, self.cond_in_dim, kernel_size=1, bias=True
        )

    def _get_pe(
        self, n_out: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        buf_name = f"_sincos_pe_{n_out}"
        if not hasattr(self, buf_name):
            pe = _get_3d_sincos_embed(n_out, self.cond_in_dim)
            self.register_buffer(buf_name, pe, persistent=False)
        return getattr(self, buf_name).to(device=device, dtype=dtype)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        Input:
            field: (B, R, R, R) or (B, C_in, R, R, R)
        Output:
            (B, R'^3, cond_in_dim) token sequence with 3D PE added.
        """
        if field.dim() == 4:
            field = field.unsqueeze(1)
        elif field.dim() != 5:
            raise ValueError(
                f"field must be 4D (B,R,R,R) or 5D (B,C,R,R,R), got {tuple(field.shape)}"
            )
        if field.shape[1] != self.in_channels:
            raise ValueError(
                f"field channel dim {field.shape[1]} != in_channels {self.in_channels}"
            )
        if not (field.shape[2] == field.shape[3] == field.shape[4]):
            raise ValueError(
                f"field must be cubic (R,R,R), got spatial {tuple(field.shape[2:])}"
            )

        x = self.stem(field)
        for blk in self.down_blocks:
            x = blk(x)
        x = self.tail_blocks(x)
        feat = self.proj(x)

        n_out = feat.shape[-1]
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        if self.pos_embed == "sincos":
            pe = self._get_pe(n_out, tokens.device, tokens.dtype)
            tokens = tokens + pe.unsqueeze(0)
        return tokens
