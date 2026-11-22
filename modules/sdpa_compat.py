"""Pure-PyTorch stand-ins for the `flash_attn` functions used in this repo.

`flash_attn` only builds against CUDA (or ROCm on a few data-center cards), which
rules out CPU-only machines and most consumer AMD / Intel GPUs. The functions
below keep the same signatures and tensor layouts but run on
`torch.nn.functional.scaled_dot_product_attention`, so they work on any device
PyTorch supports. They are slower than flash-attn on CUDA and are only selected
when flash-attn is missing or when `ATTN_BACKEND=sdpa` is set.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

# Upper bound on (batch * q_len * k_len) attention entries evaluated at once,
# to keep the padded batches from exhausting memory on long sequences.
_MAX_ENTRIES = 1 << 26


def _lengths(cu_seqlens: torch.Tensor) -> List[int]:
    cu = cu_seqlens.to(torch.int64)
    return (cu[1:] - cu[:-1]).tolist()


def _pad(x: torch.Tensor, starts: List[int], lens: List[int], max_len: int) -> torch.Tensor:
    """[T, H, C] packed tokens -> [B, H, max_len, C] zero-padded batch."""
    out = x.new_zeros((len(lens), max_len) + tuple(x.shape[1:]))
    for i, (s, n) in enumerate(zip(starts, lens)):
        out[i, :n] = x[s:s + n]
    return out.transpose(1, 2)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    **_unused,
) -> torch.Tensor:
    """q: [Tq, H, C], k: [Tk, H, C], v: [Tk, H, Co] packed; returns [Tq, H, Co]."""
    q_lens, k_lens = _lengths(cu_seqlens_q), _lengths(cu_seqlens_k)
    assert len(q_lens) == len(k_lens), "q and k must hold the same number of sequences"
    q_starts = cu_seqlens_q.to(torch.int64)[:-1].tolist()
    k_starts = cu_seqlens_k.to(torch.int64)[:-1].tolist()
    out = q.new_zeros((q.shape[0], q.shape[1], v.shape[-1]))

    # Sequences of similar length are batched together to limit padding.
    order = sorted(range(len(q_lens)), key=lambda i: (q_lens[i], k_lens[i]))
    i = 0
    while i < len(order):
        j, lq, lk = i, 0, 0
        while j < len(order):
            nq, nk = max(lq, q_lens[order[j]]), max(lk, k_lens[order[j]])
            if j > i and (j - i + 1) * nq * nk > _MAX_ENTRIES:
                break
            lq, lk, j = nq, nk, j + 1
        idx = [n for n in order[i:j] if q_lens[n] > 0]
        i = j
        if not idx or lk == 0:
            continue
        qs, ql = [q_starts[n] for n in idx], [q_lens[n] for n in idx]
        ks, kl = [k_starts[n] for n in idx], [k_lens[n] for n in idx]
        qb, kb, vb = _pad(q, qs, ql, lq), _pad(k, ks, kl, lk), _pad(v, ks, kl, lk)
        mask = None
        if min(kl) < lk:
            klen = torch.as_tensor(kl, device=q.device)
            mask = (torch.arange(lk, device=q.device)[None, :] < klen[:, None])[:, None, None, :]
        ob = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask).transpose(1, 2)
        for b, (s, n) in enumerate(zip(qs, ql)):
            out[s:s + n] = ob[b, :n]
    return out


def flash_attn_varlen_qkvpacked_func(qkv: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int = 0, **_unused) -> torch.Tensor:
    """qkv: [T, 3, H, C] packed; returns [T, H, C]."""
    q, k, v = qkv.unbind(dim=1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens)


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    **_unused,
) -> torch.Tensor:
    """q: [Tq, H, C], kv: [Tk, 2, H, C] packed; returns [Tq, H, C]."""
    k, v = kv.unbind(dim=1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k)


def flash_attn_qkvpacked_func(qkv: torch.Tensor, **_unused) -> torch.Tensor:
    """qkv: [B, N, 3, H, C] dense; returns [B, N, H, C]."""
    q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
