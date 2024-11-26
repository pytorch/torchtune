# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from torchtune.training.checkpointing._checkpointer import (
    FullModelHFCheckpointer,
    FullModelMetaCheckpointer,
    FullModelTorchTuneCheckpointer,
)
from torchtune.training.checkpointing._utils import (
    ADAPTER_CONFIG,
    ADAPTER_CONFIG_FNAME,
    ADAPTER_KEY,
    ADAPTER_MODEL_FNAME,
    EPOCHS_KEY,
    FormattedCheckpointFiles,
    MAX_STEPS_KEY,
    MODEL_KEY,
    ModelType,
    OPT_KEY,
    REPO_ID_FNAME,
    RNG_KEY,
    SAFETENSOR_INDEX_FNAME,
    SEED_KEY,
    SHARD_FNAME,
    STEPS_KEY,
    TORCHTUNE_INDEX_FNAME,
    TOTAL_EPOCHS_KEY,
    update_state_dict_for_classifier,
)

Checkpointer = Union[
    FullModelHFCheckpointer,
    FullModelMetaCheckpointer,
    FullModelTorchTuneCheckpointer,
]

__all__ = [
    "FullModelHFCheckpointer",
    "FullModelMetaCheckpointer",
    "FullModelTorchTuneCheckpointer",
    "ModelType",
    "Checkpointer",
    "update_state_dict_for_classifier",
    "ADAPTER_CONFIG",
    "ADAPTER_CONFIG_FNAME",
    "ADAPTER_MODEL_FNAME",
    "SHARD_FNAME",
    "SAFETENSOR_INDEX_FNAME",
    "TORCHTUNE_INDEX_FNAME",
    "REPO_ID_FNAME",
    "ADAPTER_KEY",
    "EPOCHS_KEY",
    "MAX_STEPS_KEY",
    "MODEL_KEY",
    "OPT_KEY",
    "RNG_KEY",
    "SEED_KEY",
    "STEPS_KEY",
    "TOTAL_EPOCHS_KEY",
    "FormattedCheckpointFiles",
]
