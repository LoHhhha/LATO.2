import torch

from utils import logging


def load_latov2_model(cls, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = cls(**ckpt["config"]["args"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    logging.info(f"loaded {ckpt.get('id', cls.__name__)} from {ckpt_path}")
    return model.to(device).eval(), ckpt["config"]
