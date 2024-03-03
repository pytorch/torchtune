# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch

from torchtune.models.llama2 import (
    convert_llama2_from_hf_format,
    convert_llama2_from_meta_format,
    convert_llama2_to_hf_format,
    convert_llama2_to_meta_format,
)
from torchtune.utils import CheckpointFormat, ModelType


def convert_to_torchtune_format(
    original_state_dict: Dict[str, torch.Tensor],
    original_ckpt_format: CheckpointFormat,
    model_type: ModelType,
) -> Dict[str, torch.Tensor]:
    """ """
    converted_state_dict = {}
    if model_type == ModelType.LLAMA2:
        converted_state_dict = _convert_llama2_to_torchtune_format(
            original_state_dict, original_ckpt_format
        )
    else:
        raise NotImplementedError(
            f"Model {model_type} is not currently supported in TorchTune."
        )
    return converted_state_dict


def _convert_llama2_to_torchtune_format(
    original_state_dict: Dict[str, torch.Tensor],
    original_ckpt_format: CheckpointFormat,
) -> Dict[str, torch.Tensor]:
    """ """
    converted_state_dict = {}
    if original_ckpt_format == CheckpointFormat.META_FORMAT:
        converted_state_dict = convert_llama2_from_meta_format(original_state_dict)
    elif original_ckpt_format == CheckpointFormat.HF_FORMAT:
        converted_state_dict = convert_llama2_from_hf_format(original_state_dict)
    else:
        raise NotImplementedError(
            f"Checkpoint format {original_ckpt_format} is not currently supported "
            "for Llama2 models in TorchTune."
        )
    return converted_state_dict


def convert_from_torchtune_format(
    state_dict: Dict[str, torch.Tensor],
    final_ckpt_format: CheckpointFormat,
    model_type: ModelType,
) -> Dict[str, torch.Tensor]:
    """ """
    if model_type == ModelType.LLAMA2:
        converted_state_dict = _convert_llama2_from_torchtune_format(
            state_dict, final_ckpt_format
        )
    else:
        raise NotImplementedError(
            f"Model {model_type} is not currently supported in TorchTune."
        )
    return converted_state_dict


def _convert_llama2_from_torchtune_format(
    state_dict: Dict[str, torch.Tensor],
    final_ckpt_format: CheckpointFormat,
) -> Dict[str, torch.Tensor]:
    """ """
    converted_state_dict = {}
    if final_ckpt_format == CheckpointFormat.META_FORMAT:
        converted_state_dict = convert_llama2_to_meta_format(state_dict)
    elif final_ckpt_format == CheckpointFormat.HF_FORMAT:
        converted_state_dict = convert_llama2_to_hf_format(state_dict)
    else:
        raise NotImplementedError(
            f"Checkpoint format {source_ckpt_format} is not currently supported "
            "for Llama2 models in TorchTune."
        )
    return converted_state_dict
