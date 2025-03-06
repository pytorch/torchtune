# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .rewards import batch_shaped_correctness_reward

DEFAULT_REWARD_FN = batch_shaped_correctness_reward