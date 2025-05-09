# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from tensordict import TensorClass
from torchtune.dev.rl.rewards import RewardOutput


class Trajectory(TensorClass["nocast"]):
    query_responses: torch.Tensor
    responses: torch.Tensor
    logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    query_response_padding_masks: torch.Tensor
    seq_lens: torch.Tensor
    answers: torch.Tensor
    policy_version: int
    advantages: torch.Tensor
    reward_outputs: List[RewardOutput]
    sequence_ids: List[str]
