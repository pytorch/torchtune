# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from ._checkpointer import (  # noqa
    FullModelHFCheckpointer,
    FullModelMetaCheckpointer,
    FullModelTorchTuneCheckpointer,
)
from ._checkpointer_utils import ModelType  # noqa


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
]
