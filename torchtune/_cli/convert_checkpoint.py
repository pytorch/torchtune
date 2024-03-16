# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This file converts download checkpoints to a format compatible with Torchtune."""

import argparse
from pathlib import Path
from typing import Optional

import torch

from torchtune.models.llama2 import convert_llama2_fair_format
from torchtune.utils import get_logger
from torchtune.utils.constants import MODEL_KEY

_PYTORCH_MODEL_FILENAME = "native_pytorch_model.pt"

log = get_logger("DEBUG")


def convert_checkpoint(
    checkpoint_path: Path,
    model: str,
    output_path: Optional[Path] = None,
    train_type: str = "full",
    output_numerical_validation: bool = False,
):
    """Convert model checkpoint to a PyTorch-native format compatible with Torchtune.

    Args:
        checkpoint_path (Path): Path to the checkpoint path.
        model (str): Model name
        output_path (Optional[Path]): Path to the output checkpoint.
        train_type (str): Type of finetuning
        output_numerical_validation (bool): Whether to run numerical validation on the converted checkpoint.

    Raises:
        Exception: If unsupported model is provided.
    """
    # Load the original state dict
    original_state_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    log.info(msg="Loaded original state dict")

    # Convert checkpoint
    if model == "llama2":
        state_dict = convert_llama2_fair_format(
            original_state_dict, output_numerical_validation
        )
    else:
        raise NotImplementedError(f"Model {model} is not supported in TorchTune.")

    # Save the state dict
    if output_path is None:
        checkpoint_dir = checkpoint_path.parent
        output_path = checkpoint_dir / _PYTORCH_MODEL_FILENAME

    output_state_dict = {}
    if train_type == "lora":
        output_state_dict[MODEL_KEY] = state_dict
    else:
        output_state_dict = state_dict
    torch.save(output_state_dict, output_path)

    log.info(msg=f"Succesfully wrote PyTorch-native model checkpoint to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path", type=Path, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Where to write the converted checkpoint. "
        "Will default to the same directory as the original checkpoint if no arg is provided"
        f"under the filename {_PYTORCH_MODEL_FILENAME}.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model name",
        choices=["llama2"],
        required=True,
    )
    parser.add_argument(
        "--train-type",
        type=str,
        help="Type of finetuning. Currently Full-Finetuning and LoRA have slightly different formats. "
        "This will be resolved soon.",
        choices=["full", "lora"],
        required=True,
    )
    parser.add_argument(
        "--output-numerical-validation",
        action="store_true",
        help="Whether to load the original checkpoint and the converted checkpoint and compare"
        "the numerical output of a forward pass to ensure that the conversion was successful."
        "Prints results to stdout. This additional check is only available for Llama2 7B."
        "This will take awhile and may consume lots of memory. If you see an OOM error,"
        "please disable this flag. Note: All our checkpoints conversions are already validated"
        "in unit tests for smaller checkpoints and integration tests for larger checkpoints."
        "This flag is primarily for debugging purposes.",
        required=False,
        default=False,
    )
    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint_path,
        args.model,
        args.output_path,
        args.train_type,
        args.output_numerical_validation,
    )
