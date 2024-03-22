# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This script downloads a model from the HuggingFace hub."""

from huggingface_hub import snapshot_download


def download_cmd(*args) -> None:
    """Downloads a model from the Hugging Face Hub.

    Raises:
        ValueError: If the model is not supported.
    """
    if "meta-llama" in args.repo_id and args.hf_token is None:
        raise ValueError(
            "You need to provide a Hugging Face API token to download gated models."
            "You can find your token by visiting https://huggingface.co/settings/tokens"
        )

    # Download the tokenizer and PyTorch model files
    snapshot_download(
        args.repo_id,
        local_dir=args.output_dir,
        resume_download=True,
        token=args.hf_token,
    )

    print(
        "Succesfully downloaded model repo and wrote to the following locations:",
        *list(args.output_dir.iterdir()),
        sep="\n",
    )
