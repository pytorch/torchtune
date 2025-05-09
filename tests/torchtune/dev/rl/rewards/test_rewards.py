# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtune.dev.rl.rewards import RewardOutput


class TestRewardOutput:
    @pytest.fixture
    def sample_reward_output(self):
        return RewardOutput(
            reward_base_name="test_reward",
            total_reward=torch.tensor([1.0, 2.0, 3.0]),
            successes=torch.tensor([1.0, 0.0, 1.0]),
            rewards={
                "sub_reward_1": torch.tensor([0.5, 1.5, 2.5]),
                "sub_reward_2": torch.tensor([10.0, 20.0, 30.0]),
            },
        )

    def test_log(self, sample_reward_output):
        log_dict = sample_reward_output.log(prefix="train")
        expected_log = {
            "train/test_reward/sub_reward_1": 1.5,
            "train/test_reward/sub_reward_2": 20.0,
            "train/test_reward": 2.0,
            "train/test_reward/successes": 2.0 / 3.0,
        }
        assert log_dict.keys() == expected_log.keys()
        for key in expected_log:
            assert log_dict[key] == pytest.approx(expected_log[key])

    def test_log_no_prefix(self, sample_reward_output):
        log_dict = sample_reward_output.log()
        expected_log = {
            "test_reward/sub_reward_1": 1.5,
            "test_reward/sub_reward_2": 20.0,
            "test_reward": 2.0,
            "test_reward/successes": 2.0 / 3.0,
        }
        assert log_dict.keys() == expected_log.keys()
        for key in expected_log:
            assert log_dict[key] == pytest.approx(expected_log[key])
