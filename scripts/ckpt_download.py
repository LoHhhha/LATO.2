"""
Usage:
    python scripts/ckpt_download.py \
        [--out_dir <path>]
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils import logging

DEFAULT_REPO_ID = "0x4c48/LATO.2"
DEFAULT_OUT_DIR = os.path.join(ROOT, "ckpt")


def parse_args():
    p = argparse.ArgumentParser(
        description="Download LATO.2 checkpoints from the Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo_id", default=DEFAULT_REPO_ID, help="HF model repo id")
    p.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help="Directory to download checkpoints into",
    )
    p.add_argument("--revision", default=None, help="Git revision / branch / tag")
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF access token for gated/private repos (or set HF_TOKEN)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Download every file in the repo (not just *.pt)",
    )
    p.add_argument(
        "--include-readme",
        action="store_true",
        help="Also download README.md alongside the *.pt weights",
    )
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed. Activate the `trellis2` conda env "
            "or run: pip install -U huggingface_hub"
        )

    if args.all:
        allow_patterns = None
    else:
        allow_patterns = ["*.pt"]
        if args.include_readme:
            allow_patterns.append("README.md")

    os.makedirs(args.out_dir, exist_ok=True)
    logging.info(f"Downloading {args.repo_id} -> {args.out_dir}")
    if allow_patterns:
        logging.info(f"  patterns: {allow_patterns}")

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=args.out_dir,
        allow_patterns=allow_patterns,
        token=args.token,
    )

    logging.info(f"\nDone. Checkpoints available in: {path}")
    files = sorted(f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
    for f in files:
        size = os.path.getsize(os.path.join(path, f)) / (1024 * 1024)
        logging.info(f"  {f:24s} {size:8.1f} MB")


if __name__ == "__main__":
    main()
