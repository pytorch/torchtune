# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ._logging import deprecated, get_logger, log_once, log_rank_zero

from ._version import torch_version_ge

__all__ = [
    "get_logger",
    "torch_version_ge",
    "log_rank_zero",
    "deprecated",
    "log_once",
]
