import torch
import torch.nn as nn


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        variance_in_fp32: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.variance_in_fp32 = bool(variance_in_fp32)
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if self.variance_in_fp32:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            inv_rms = torch.rsqrt(variance + self.eps).to(input_dtype)
        else:
            variance = x.pow(2).mean(-1, keepdim=True)
            inv_rms = torch.rsqrt(variance + self.eps)
        x = x * inv_rms
        if self.weight is not None:
            w = self.weight
            if w.dtype in (torch.float16, torch.bfloat16):
                x = (x.to(w.dtype) * w).to(input_dtype)
            else:
                x = (x * w).to(input_dtype)
        else:
            x = x.to(input_dtype)
        return x
