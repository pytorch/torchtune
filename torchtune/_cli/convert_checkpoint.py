# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This file converts download checkpoints to a format compatible with Torchtune."""

from pathlib import Path
from typing import Optional

import torch

from torchtune.models.llama2 import convert_llama2_fair_format
from torchtune.utils import get_logger
from torchtune.utils.constants import MODEL_KEY

_PYTORCH_MODEL_FILENAME = "native_pytorch_model.pt"

log = get_logger("DEBUG")


def convert_checkpoint_cmd(
    args
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
    checkpoint_path = args.checkpoint_path
    model = args.model
    output_path = args.output_path
    train_type = args.train_type
    output_numerical_validation = args.output_numerical_validation

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
