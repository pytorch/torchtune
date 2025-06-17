# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from torchtune.utils._device import has_cuda_capability
from torchtune.utils._logging import get_logger, log_once

_log: logging.Logger = get_logger()

# Configuration of MoE
# use grouped_mm in MoE or for loop for experts computation.
use_grouped_mm = True


def should_use_grouped_mm():
    if use_grouped_mm and not has_cuda_capability(9, 0):
        log_once(
            _log,
            "Failed to use grouped mm, which is only supported on SM90 or later",
            level=logging.DEBUG,
        )
        return False
    return use_grouped_mm
