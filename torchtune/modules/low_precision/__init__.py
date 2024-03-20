# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch.nn as nn
from torchao.dtypes.nf4tensor import NF4Tensor

from ._state_dict_hooks import reparametrize_as_bf16_state_dict_post_hook

from .nf4_linear import FrozenNF4Linear

__all__ = ["FrozenNF4Linear", "reparametrize_as_bf16_state_dict_post_hook"]
