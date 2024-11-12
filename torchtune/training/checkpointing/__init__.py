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
    ADAPTER_KEY,
    EPOCHS_KEY,
    FormattedCheckpointFiles,
    MAX_STEPS_KEY,
    MODEL_KEY,
    ModelType,
    OPT_KEY,
    RNG_KEY,
    SEED_KEY,
    STEPS_KEY,
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
