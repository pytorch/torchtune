# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ._device import get_device

from ._version import torch_version_ge
from .logging import get_logger

__all__ = [
    "get_device",
    "get_logger",
    "torch_version_ge",
]
