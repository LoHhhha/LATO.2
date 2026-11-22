import contextlib

import torch


def add_device_arg(parser):
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device: 'auto' (cuda if available, else cpu), 'cuda', 'cpu', ...",
    )


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def autocast(device: torch.device):
    """bfloat16 autocast on CUDA; full precision elsewhere (slow or unsupported in bf16)."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
