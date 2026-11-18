"""TRELLIS stage-1 (sparse structure) models: image -> 64^3 occupancy.

Self-contained port of the sparse-structure flow and decoder of
https://github.com/microsoft/TRELLIS, enough to load the released checkpoints.
"""

import json
import os

from safetensors.torch import load_file

from utils import logging

from .decoder import SparseStructureDecoder
from .flow import SparseStructureFlowModel
from .sampler import SSFlowEulerSampler

TRELLIS_REPO_ID = "microsoft/TRELLIS-image-large"
SS_FLOW_NAME = "ss_flow_img_dit_L_16l8_fp16"
SS_DECODER_NAME = "ss_dec_conv3d_16l8_fp16"
# image conditioning the released SS flow was trained with
SS_COND_MODEL = "dinov2_vitl14_reg"
SS_COND_RESOLUTION = 518

_MODEL_CLASSES = {
    "SparseStructureFlowModel": SparseStructureFlowModel,
    "SparseStructureDecoder": SparseStructureDecoder,
}


def load_trellis_model(name, ckpt_dir, device, repo_id=TRELLIS_REPO_ID):
    """Load a TRELLIS checkpoint pair ``ckpts/<name>.json`` + ``.safetensors``.

    Files already under ``ckpt_dir`` are reused; missing ones are downloaded
    there from the Hugging Face Hub repo ``repo_id``.
    """
    paths = {}
    for ext in ("json", "safetensors"):
        rel = f"ckpts/{name}.{ext}"
        path = os.path.join(ckpt_dir, rel)
        if not os.path.isfile(path):
            from huggingface_hub import hf_hub_download

            logging.info(f"{path} not found; downloading {repo_id}/{rel}")
            path = hf_hub_download(repo_id, rel, local_dir=ckpt_dir)
        paths[ext] = path

    with open(paths["json"]) as f:
        config = json.load(f)
    if config["name"] not in _MODEL_CLASSES:
        raise ValueError(
            f"{paths['json']}: unsupported TRELLIS model {config['name']!r}; "
            f"supported: {sorted(_MODEL_CLASSES)}"
        )
    model = _MODEL_CLASSES[config["name"]](**config["args"])
    model.load_state_dict(load_file(paths["safetensors"]), strict=True)
    logging.info(f"loaded {config['name']} from {paths['safetensors']}")
    return model.to(device).eval()
