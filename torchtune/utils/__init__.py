# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ._device import (
    batch_to_device,
    DeviceSupport,
    get_device,
    get_device_support,
    get_torch_device,
    is_npu_available,
)
from ._logging import get_logger

from ._version import torch_version_ge

__all__ = [
    "batch_to_device",
    "get_device",
    "get_logger",
    "torch_version_ge",
    "is_npu_available",
    "get_device_support",
    "get_torch_device",
    "DeviceSupport",
]
