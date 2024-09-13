# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Union

import torch

CROSS_ENTROPY_IGNORE_IDX = -100
PACK_TYPE = Dict[str, Union[torch.Tensor, List[int]]]
