from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import convert_module_to_f16, convert_module_to_f32
from modules.transformer import (
    AbsolutePositionEmbedder,
    TimestepEmbedder,
)
from modules import sparse as sp
from modules.sparse.transformer import ModulatedSparseTransformerCrossBlock


class VertexSLatFlowModel(nn.Module):
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
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        use_density: bool = False,
        **kwargs
    ):
        if kwargs:
            print(f"[SLatFlowModel] Found unused arguments: {kwargs}")
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.use_density = use_density
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)

        if self.use_density:
            self.density_embedder = TimestepEmbedder(model_channels)

        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(
            in_channels,
            model_channels,
        )

        self.blocks = nn.ModuleList(
            [
                ModulatedSparseTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode="full",
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    share_mod=self.share_mod,
                    qk_rms_norm=self.qk_rms_norm,
                    qk_rms_norm_cross=self.qk_rms_norm_cross,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.out_layer = sp.SparseLinear(
            model_channels,
            out_channels,
        )

        if use_fp16:
            self.convert_to_fp16()
        else:
            self.convert_to_fp32()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        density: Optional[torch.Tensor] = None,
    ) -> sp.SparseTensor:
        h = self.input_layer(x).type(self.dtype)
        t_emb = self.t_embedder(t)
        if self.use_density:
            assert (
                density is not None
            ), "Density tensor must be provided when use_density is True"
            t_emb = t_emb + self.density_embedder(density.reshape(-1).float())
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(x.dtype))
        return h
