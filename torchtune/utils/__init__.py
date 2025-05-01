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
    get_torch_device_namespace,
    get_world_size_and_rank,
)
from ._logging import deprecated, get_logger, log_once, log_rank_zero

from ._version import torch_version_ge

__all__ = [
    "get_world_size_and_rank",
    "batch_to_device",
    "get_device",
    "get_logger",
    "torch_version_ge",
    "get_device_support",
    "get_torch_device_namespace",
    "DeviceSupport",
    "log_rank_zero",
    "deprecated",
    "log_once",
]
