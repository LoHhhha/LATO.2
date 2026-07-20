import torch
import torch.nn as nn
from typing import *
import numpy as np
from modules import sparse as sp

FP16_MODULES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
    sp.SparseConv3d,
    sp.SparseInverseConv3d,
    sp.SparseLinear,
)


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, FP16_MODULES):
        for p in l.parameters():
            p.data = p.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, FP16_MODULES):
        for p in l.parameters():
            p.data = p.data.float()


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiagonalGaussianDistribution(object):
    def __init__(
        self,
        parameters: Union[torch.Tensor, List[torch.Tensor]],
        deterministic=False,
        feat_dim=1,
    ):
        self.feat_dim = feat_dim
        self.parameters = parameters

        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.mean(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=dims
                )
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=dims,
                )

    def nll(self, sample, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


def per_batch_counts(batch_indices: torch.Tensor, num_batches: int) -> List[int]:
    """Count elements per batch, returned as a list of length num_batches."""
    return torch.bincount(batch_indices.long(), minlength=num_batches).tolist()


def flatten_coords(coords_4d: torch.Tensor):
    coords_4d_long = coords_4d.long()

    base_x = 1024
    base_y = 1024 * 1024
    base_z = 1024 * 1024 * 1024

    flat_coords = (
        coords_4d_long[:, 0] * base_z
        + coords_4d_long[:, 1] * base_y
        + coords_4d_long[:, 2] * base_x
        + coords_4d_long[:, 3]
    )
    return flat_coords

def manual_cast(tensor, dtype):
    if not torch.is_autocast_enabled():
        return tensor.type(dtype)
    return tensor


def str_to_dtype(dtype_str: str):
    return {
        "f16": torch.float16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "f32": torch.float32,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[dtype_str]