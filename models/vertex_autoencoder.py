import torch
import torch.nn as nn
from typing import *
import torch.nn.functional as F

from modules import sparse as sp
from modules.sparse import SparseTensor
from modules.sparse.linear import SparseLinear
from modules.sparse.nonlinearity import SparseGELU
from modules.utils import (
    zero_module,
    convert_module_to_f16,
    convert_module_to_f32,
    flatten_coords,
    per_batch_counts,
)
from modules.sparse.transformer import SparseTransformerBase, SparseTransformerCrossBase
from modules.sparse.blocks import SparseResBlock3d
from modules.utils import DiagonalGaussianDistribution


class SparseOccHead(nn.Module):
    def __init__(self, channels: int, out_channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            SparseLinear(channels, int(channels * mlp_ratio)),
            SparseGELU(approximate="tanh"),
            SparseLinear(int(channels * mlp_ratio), out_channels),
        )

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        return self.mlp(x)


class SparseEncoderBlock(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_downsample: int = 4,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.resolution = resolution

        self.self_attn = SparseTransformerBase(
            in_channels=model_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            mlp_ratio=mlp_ratio,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )

        self.input_layer1 = sp.SparseLinear(
            in_channels, model_channels >> num_downsample
        )

        self.downsample = nn.ModuleList(
            [
                SparseResBlock3d(
                    channels=model_channels >> (i + 1),
                    out_channels=model_channels >> i,
                    downsample=True,
                    upsample=False,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(num_downsample - 1, -1, -1)
            ]
        )

    def forward(
        self,
        x: SparseTensor,
    ):
        """
        Input:
            x: SparseTensor in N resolution, with feats of in_channels
        Output:
            h: SparseTensor in N>>num_downsample resolution, with feats of model_channels
        """
        x = self.input_layer1(x)
        for block in self.downsample:
            x = block(x)
        h = self.self_attn(x)
        return h


class SparseDecoderUpsampleBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: int,
        model_channels: int = 512,
        num_blocks: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_groups: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.out_resolution = resolution * 2
        self.model_channels = model_channels
        self.out_channels = out_channels

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels), sp.SparseSiLU()
        )

        self.sub = sp.SparseSubdivide()

        self.out_layers = nn.Sequential(
            sp.SparseConv3d(
                channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"
            ),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            zero_module(
                sp.SparseConv3d(
                    self.out_channels,
                    self.out_channels,
                    3,
                    indice_key=f"res_{self.out_resolution}",
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(
                channels, self.out_channels, 1, indice_key=f"res_{self.out_resolution}"
            )

        self.pruning_head = SparseOccHead(self.out_channels, out_channels=1)

        self.ca = SparseTransformerCrossBase(
            in_channels=self.out_channels,
            model_channels=self.model_channels,
            context_channels=self.model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_mode="full",
            pe_mode="ape",
            use_checkpoint=True,
            qk_rms_norm=False,
        )

        self.proj_ctx = sp.SparseLinear(self.out_channels, self.model_channels)
        self.proj_out = sp.SparseLinear(self.model_channels, self.out_channels)

    def forward(
        self,
        x: sp.SparseTensor,
        training=False,
        threshold=0.5,
    ) -> sp.SparseTensor:
        h = self.act_layers(x)
        h = self.sub(h)
        x_sub = self.sub(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x_sub)
        h = self.proj_out(self.ca(x=h, context=self.proj_ctx(h)))

        occ_prob_q = self.pruning_head(h)

        if training:
            return h, occ_prob_q, [0]

        scores_q = torch.sigmoid(occ_prob_q.feats).squeeze(-1)
        N_full = h.feats.shape[0]
        if N_full % 8 != 0:
            raise ValueError(f"Number of nodes({N_full}) is not divisible by 8.")

        # ensure at least one point is kept in each group of 8
        n_parents = N_full // 8

        scores_q_grouped = scores_q.view(n_parents, 8)

        mask_grouped = scores_q_grouped >= threshold

        none_survived = mask_grouped.sum(dim=1) == 0

        # per-batch rescue counts; all 8 children of a parent share one batch index
        if n_parents > 0:
            parent_batch = h.coords[:, 0].view(n_parents, 8)[:, 0]
            num_rescue = per_batch_counts(
                parent_batch[none_survived], int(parent_batch.max().item()) + 1
            )
        else:
            num_rescue = [0]
        if none_survived.any():
            failed_scores = scores_q_grouped[none_survived]
            _, topk_indices = torch.topk(failed_scores, k=1, dim=1)

            failed_row_idxs = torch.nonzero(none_survived, as_tuple=True)[0]
            rows_expanded = failed_row_idxs.unsqueeze(1).expand(-1, 1)

            mask_grouped[rows_expanded, topk_indices] = True

        sub_mask = mask_grouped.view(-1)

        h = sp.SparseTensor(feats=h.feats[sub_mask], coords=h.coords[sub_mask])
        occ_prob_final = sp.SparseTensor(
            feats=occ_prob_q.feats[sub_mask], coords=occ_prob_q.coords[sub_mask]
        )

        return h, occ_prob_final, num_rescue


class SparseDecoderBlock(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        out_channels: int,
        model_channels: int = 512,
        num_blocks: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
    ):
        super().__init__()
        self.resolution = resolution

        self.upsample = SparseDecoderUpsampleBlock(
            channels=in_channels,
            resolution=resolution,
            out_channels=out_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            model_channels=model_channels,
            num_groups=32,
        )

        if use_fp16:
            self.convert_to_fp16()

    def forward(
        self,
        x: sp.SparseTensor,
        training: bool = False,
        threshold: float = 0.5,
    ):
        h = x
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h, occ_prob, num_rescue = self.upsample(
            h,
            training=training,
            threshold=threshold,
        )
        return h, occ_prob, num_rescue

    def convert_to_fp16(self):
        """Convert all components to float16"""
        convert_module_to_f16(self.upsample)

    def convert_to_fp32(self):
        """Convert all components to float32"""
        convert_module_to_f32(self.upsample)


class VertexVAE(nn.Module):
    def __init__(
        self,
        # Core architecture parameters
        encoder_cfg: Dict = {},
        expander_cfg: Dict = {},
        decoder_cfg: List[Dict] = [],
        # Shared transformer parameters
        resolution: int = 1024,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        attn_mode: str = "swin",
        window_size: int = 8,
        pe_mode: str = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = True,
        qk_rms_norm: bool = False,
        latent_dim: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder_cfg = decoder_cfg

        self.encoder = SparseEncoderBlock(
            resolution=resolution,
            in_channels=encoder_cfg["in_channels"],
            model_channels=encoder_cfg["model_channels"],
            num_blocks=encoder_cfg["num_blocks"],
            num_heads=encoder_cfg["num_heads"],
            num_downsample=len(decoder_cfg),
            num_head_channels=num_head_channels,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            mlp_ratio=mlp_ratio,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )

        self.latent_expander = SparseTransformerBase(
            in_channels=latent_dim,
            model_channels=expander_cfg["model_channels"],
            num_blocks=expander_cfg["num_blocks"],
            num_heads=expander_cfg["num_heads"],
            num_head_channels=num_head_channels,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            mlp_ratio=mlp_ratio,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )

        self.vtx_proj = sp.SparseLinear(
            expander_cfg["model_channels"], decoder_cfg[0]["in_channels"]
        )

        self.vtx_pruning_head = SparseOccHead(
            expander_cfg["model_channels"], out_channels=1
        )

        self.out_layer = sp.SparseLinear(expander_cfg["model_channels"], latent_dim * 2)

        self.decoder_vtx = nn.ModuleList()
        self.decoder_vtx_ca = nn.ModuleList()
        self.latent_proj = nn.ModuleList()
        for config in decoder_cfg:
            self.decoder_vtx.append(
                # using default parameters to init the upsample block
                SparseDecoderBlock(
                    resolution=config["resolution"],
                    in_channels=config["in_channels"],
                    out_channels=config["out_channels"],
                    num_blocks=config["num_blocks"],
                    num_heads=config["num_heads"],
                    use_fp16=use_fp16,
                )
            )
            self.latent_proj.append(
                sp.SparseLinear(latent_dim, config["context_channels"])
            )
            self.decoder_vtx_ca.append(
                SparseTransformerCrossBase(
                    in_channels=config["out_channels"],
                    model_channels=config["model_channels"],
                    context_channels=config["context_channels"],
                    num_blocks=config["num_blocks"],
                    num_heads=config["num_heads"],
                    num_head_channels=num_head_channels,
                    mlp_ratio=mlp_ratio,
                    attn_mode="full",
                    window_size=window_size,
                    pe_mode=pe_mode,
                    use_fp16=use_fp16,
                    use_checkpoint=use_checkpoint,
                    qk_rms_norm=qk_rms_norm,
                )
            )

        if use_fp16:
            self.convert_to_fp16()

    def encode(
        self,
        x: sp.SparseTensor,
        sample_posterior=True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)

        posterior = DiagonalGaussianDistribution(h.feats, feat_dim=-1)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = h.replace(z)
        return z, posterior

    def decode(
        self,
        latent_: sp.SparseTensor,
        gt_vertex_voxels_list: List[sp.SparseTensor],
        training=True,
        inference_threshold=0.5,
        verbose=False,
    ) -> List[Dict]:
        """
        Args:
            latent: Initial SparseTensor from encoder at 64-resolution.
            gt_vertex_voxels_list: Ground-truth vertex SparseTensors at [64, 128, 256, 512, 1024]
            training: Whether to apply pruning during training

        Returns:
            List[Dict] with separate vertex and edge predictions at each level
        """
        latent = self.latent_expander(latent_)

        results = []

        # step0: shell voxels to vertex voxels
        vtx_probs = self.vtx_pruning_head(latent)  # (N, 1)
        if not training:
            # Inference path: use predicted vertex mask to split vertex

            scores = torch.sigmoid(vtx_probs.feats).squeeze(-1)  # (N,)

            vertex_mask = scores >= inference_threshold  # (N,)
            batch_indices = latent.coords[:, 0]
            for b in batch_indices.unique():
                batch_sel = batch_indices == b
                if vertex_mask[batch_sel].any():
                    continue
                batch_scores = scores[batch_sel]
                k = min(2, batch_scores.numel())
                print(
                    f"[VertexVAE] Warning: No points passed threshold {inference_threshold} in batch {b.item()}. Forcing top {k} points."
                )

                _, top_local = torch.topk(batch_scores, k=k)

                vertex_mask[batch_sel.nonzero(as_tuple=True)[0][top_local]] = True

            vertex_x = sp.SparseTensor(
                feats=latent.feats[vertex_mask],
                coords=latent.coords[vertex_mask],
            )

            if verbose:
                num_batches = int(latent.coords[:, 0].max().item()) + 1
                print(
                    f"[VertexVAE] Shell2Vertex: "
                    f"num_vertex={per_batch_counts(vertex_x.coords[:, 0], num_batches)}, "
                    f"num_shell={per_batch_counts(latent.coords[:, 0], num_batches)}"
                )

            results.append(
                {
                    "coords": vtx_probs.coords,
                    "occ_probs": vtx_probs.feats,
                    "vertex_mask": vertex_mask,
                }
            )
        else:
            # Training path: using gt voxels to split vertex
            gt_vertex_coords = gt_vertex_voxels_list[0].coords

            pred_flat = flatten_coords(latent.coords)
            vertex_gt_flat = flatten_coords(gt_vertex_coords)

            vertex_mask = torch.isin(pred_flat, vertex_gt_flat)

            vertex_x = sp.SparseTensor(
                feats=latent.feats[vertex_mask],
                coords=latent.coords[vertex_mask],
            )

            results.append(
                {
                    "coords": vtx_probs.coords,
                    "occ_probs": vtx_probs.feats,
                    "vertex_mask": vertex_mask,
                    "vertex_gt_coords": gt_vertex_coords,
                }
            )

        vertex_x = self.vtx_proj(vertex_x)

        # step1: upsample
        for i, _ in enumerate(self.decoder_vtx):
            vertex_x, vertex_occ_probs, num_rescue = self.decoder_vtx[i](
                vertex_x,
                training=training,
                threshold=inference_threshold,
            )
            vertex_x = self.decoder_vtx_ca[i](
                x=vertex_x,
                context=self.latent_proj[i](latent_),
            )

            if not training:
                # Inference path
                if verbose:
                    num_batches = int(latent_.coords[:, 0].max().item()) + 1
                    print(
                        f"[VertexVAE] Layer{i}: "
                        f"num_vertex={per_batch_counts(vertex_x.coords[:, 0], num_batches)}, "
                        f"num_rescue={num_rescue}"
                    )

                results.append(
                    {
                        "coords": vertex_x.coords,
                        "feats": vertex_x.feats,
                        "occ_probs": vertex_occ_probs.feats,
                        "occ_coords": vertex_occ_probs.coords,
                    }
                )
            else:
                # Training path
                vertex_pred_coords = vertex_x.coords
                gt_vertex_coords = gt_vertex_voxels_list[i + 1].coords

                vertex_pred_flat = flatten_coords(vertex_pred_coords)
                vertex_gt_flat = flatten_coords(gt_vertex_coords)
                vertex_mask = torch.isin(vertex_pred_flat, vertex_gt_flat)
                vertex_prune_labels = vertex_mask.float()

                vertex_x = sp.SparseTensor(
                    feats=vertex_x.feats[vertex_mask],
                    coords=vertex_x.coords[vertex_mask],
                )

                results.append(
                    {
                        "coords": vertex_x.coords,
                        "feats": vertex_x.feats,
                        "occ_probs": vertex_occ_probs.feats,
                        "occ_coords": vertex_occ_probs.coords,
                        "prune_labels": vertex_prune_labels,
                        "sp_tensor": vertex_x,
                        "gt_coords": gt_vertex_coords,
                        "pred_mask": vertex_mask,
                    },
                )

        return results

    def forward(
        self,
        sparse_input,
        gt_vertex_voxels_list=None,
        training=True,
        sample_posterior=True,
    ):
        latent_64, posterior = self.encode(sparse_input, sample_posterior)
        results = self.decode(
            latent_64,
            gt_vertex_voxels_list=gt_vertex_voxels_list,
            training=training,
        )

        return results, posterior, latent_64

    def convert_to_fp16(self):
        """Convert all components to float16"""
        self.encoder.apply(
            lambda m: m.convert_to_fp16() if hasattr(m, "convert_to_fp16") else None
        )
        self.decoder_vtx.apply(
            lambda m: m.convert_to_fp16() if hasattr(m, "convert_to_fp16") else None
        )
        self.decoder_vtx_ca.apply(
            lambda m: m.convert_to_fp16() if hasattr(m, "convert_to_fp16") else None
        )

    def convert_to_fp32(self):
        """Convert all components to float32"""
        self.encoder.apply(
            lambda m: m.convert_to_fp32() if hasattr(m, "convert_to_fp32") else None
        )
        self.decoder_vtx.apply(
            lambda m: m.convert_to_fp32() if hasattr(m, "convert_to_fp32") else None
        )
        self.decoder_vtx_ca.apply(
            lambda m: m.convert_to_fp32() if hasattr(m, "convert_to_fp32") else None
        )
