# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This script downloads a model from the HuggingFace hub."""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def download(repo_id: str, output_dir: Path, hf_token: str) -> None:
    """Downloads a model from the Hugging Face Hub.

    Args:
        repo_id (str): Name of the repository on Hugging Face Hub.
        output_dir (Path): Directory in which to save the model.
        hf_token (str): Hugging Face API token.

    Raises:
        ValueError: If the model is not supported.
    """
    if "meta-llama" in repo_id and hf_token is None:
        raise ValueError(
            "You need to provide a Hugging Face API token to download gated models."
            "You can find your token by visiting https://huggingface.co/settings/tokens"
        )

    # Download the tokenizer and PyTorch model files
    snapshot_download(
        repo_id,
        local_dir=output_dir,
        resume_download=True,
        token=hf_token,
    )

    print(
        "Succesfully downloaded model repo and wrote to the following locations:",
        *list(output_dir.iterdir()),
        sep="\n",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a model from the Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of the repository on Hugging Face Hub.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=False,
        default="/tmp/model",
        help="Directory in which to save the model.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        required=False,
        default=os.getenv("HF_TOKEN", None),
        help="Hugging Face API token. Needed for gated models like Llama2.",
    )
    args = parser.parse_args()
    download(args.repo_id, args.output_dir, args.hf_token)
