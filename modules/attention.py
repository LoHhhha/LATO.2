from __future__ import annotations

from typing import Optional
import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func

    _FLASH_ATTN_AVAILABLE = True
except Exception:
    flash_attn_varlen_func = None
    _FLASH_ATTN_AVAILABLE = False


def flash_varlen_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    x_mask: torch.Tensor,
) -> torch.Tensor:
    """q,k,v: (B, H, N, Dh); x_mask: (B, N) bool."""
    bsz, nheads, seqlen, head_dim = q.shape
    mask = x_mask.bool()
    lengths = mask.sum(dim=-1, dtype=torch.int32)
    if int(lengths.max().item()) <= 0:
        return torch.zeros_like(q)

    cu_seqlens = torch.zeros((bsz + 1,), dtype=torch.int32, device=q.device)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    max_seqlen = int(lengths.max().item())

    q_flat = q.permute(0, 2, 1, 3).reshape(bsz * seqlen, nheads, head_dim)
    k_flat = k.permute(0, 2, 1, 3).reshape(bsz * seqlen, nheads, head_dim)
    v_flat = v.permute(0, 2, 1, 3).reshape(bsz * seqlen, nheads, head_dim)
    valid_token_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)

    q_unpad = q_flat.index_select(0, valid_token_indices)
    k_unpad = k_flat.index_select(0, valid_token_indices)
    v_unpad = v_flat.index_select(0, valid_token_indices)

    attn_unpad = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        causal=False,
    )
    out_flat = torch.zeros_like(q_flat)
    out_flat.index_copy_(0, valid_token_indices, attn_unpad)
    out = out_flat.reshape(bsz, seqlen, nheads, head_dim).permute(0, 2, 1, 3)
    return out


def flash_varlen_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_mask: torch.Tensor,
    k_mask: torch.Tensor,
) -> torch.Tensor:
    """Varlen cross-attn. q: (B,H,Nq,Dh), k/v: (B,H,Nk,Dh), masks (B,Nq)/(B,Nk) bool."""
    bsz, nheads, nq, head_dim = q.shape
    nk = k.shape[2]
    q_mask_b = q_mask.bool()
    k_mask_b = k_mask.bool()
    q_lengths = q_mask_b.sum(dim=-1, dtype=torch.int32)
    k_lengths = k_mask_b.sum(dim=-1, dtype=torch.int32)
    if int(q_lengths.max().item()) <= 0:
        return torch.zeros_like(q)

    cu_q = torch.zeros((bsz + 1,), dtype=torch.int32, device=q.device)
    cu_q[1:] = torch.cumsum(q_lengths, dim=0)
    cu_k = torch.zeros((bsz + 1,), dtype=torch.int32, device=q.device)
    cu_k[1:] = torch.cumsum(k_lengths, dim=0)
    max_q = int(q_lengths.max().item())
    max_k = int(k_lengths.max().item())

    q_flat = q.permute(0, 2, 1, 3).reshape(bsz * nq, nheads, head_dim)
    k_flat = k.permute(0, 2, 1, 3).reshape(bsz * nk, nheads, head_dim)
    v_flat = v.permute(0, 2, 1, 3).reshape(bsz * nk, nheads, head_dim)

    q_idx = torch.nonzero(q_mask_b.reshape(-1), as_tuple=False).squeeze(-1)
    k_idx = torch.nonzero(k_mask_b.reshape(-1), as_tuple=False).squeeze(-1)

    q_unpad = q_flat.index_select(0, q_idx)
    k_unpad = k_flat.index_select(0, k_idx)
    v_unpad = v_flat.index_select(0, k_idx)

    attn_unpad = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        dropout_p=0.0,
        causal=False,
    )
    out_flat = torch.zeros_like(q_flat)
    out_flat.index_copy_(0, q_idx, attn_unpad)
    out = out_flat.reshape(bsz, nq, nheads, head_dim).permute(0, 2, 1, 3)
    return out


def can_flash_varlen(q: torch.Tensor, x_mask: Optional[torch.Tensor]) -> bool:
    if not _FLASH_ATTN_AVAILABLE or x_mask is None:
        return False
    if not q.is_cuda:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    return True


def sdpa_padding_mask(x_mask: torch.Tensor) -> torch.Tensor:
    """(B, 1, 1, N) bool: keys valid."""
    return x_mask.bool().view(x_mask.shape[0], 1, 1, x_mask.shape[1])


def graph_adj_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    x_mask: torch.Tensor,
    adj_matrix: Optional[torch.Tensor],
) -> torch.Tensor:
    bsz, nheads, seqlen, _ = q.shape
    out = torch.zeros_like(q)
    x_mask = x_mask.bool()

    for b in range(bsz):
        valid_indices = torch.nonzero(x_mask[b], as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            continue
        q_b = q[b].index_select(1, valid_indices).unsqueeze(0)
        k_b = k[b].index_select(1, valid_indices).unsqueeze(0)
        v_b = v[b].index_select(1, valid_indices).unsqueeze(0)
        l_now = valid_indices.numel()

        if adj_matrix is not None:
            sub = (
                adj_matrix[b]
                .bool()
                .index_select(0, valid_indices)
                .index_select(1, valid_indices)
            )
            eye = torch.eye(l_now, dtype=torch.bool, device=q.device)
            attn_mask_b = (sub | eye).view(1, 1, l_now, l_now)
        else:
            attn_mask_b = None

        out_b = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=attn_mask_b)
        out[b].index_copy_(1, valid_indices, out_b.squeeze(0))
    return out
