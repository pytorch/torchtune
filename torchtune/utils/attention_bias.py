# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
from torch import Tensor


def sample_packing_block_causal_mask(document_ids: Tensor) -> Callable:
    def score_mod(score, b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[q_idx] == document_ids[kv_idx]
        return torch.where(causal_mask & document_mask, score, -float("inf"))

    return score_mod
