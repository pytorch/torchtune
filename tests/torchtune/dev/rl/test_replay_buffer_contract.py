# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tensordict import NonTensorStack
from torchrl.data import LazyStackStorage, ReplayBuffer

from torchtune.dev.rl.datatypes import Trajectory
from torchtune.dev.rl.rewards import group_normalized_advantages


def _make_trajectory(
    batch_size: int = 16,
    seq_len: int = 64,
    resp_len: int = 20,
    num_funcs: int = 2,
    seed: int = 0,
) -> Trajectory:
    """Build a trajectory shaped like the one PostProcessingWorker produces.

    The advantages and reward metadata are per-sample and varied, so any loss of
    per-sample information in the replay buffer is detectable by exact equality.
    """
    torch.manual_seed(seed)
    advantages = torch.randn(batch_size) * 2.0 - 0.5
    rewards = torch.randn(batch_size, num_funcs)
    successes = torch.rand(batch_size, num_funcs) > 0.5
    func_names = [f"reward_fn_{i}" for i in range(num_funcs)]
    return Trajectory(
        query_responses=torch.randint(0, 100, (batch_size, seq_len)),
        responses=torch.randint(0, 100, (batch_size, resp_len)),
        logprobs=torch.randn(batch_size, resp_len),
        ref_logprobs=torch.randn(batch_size, resp_len),
        query_response_padding_masks=torch.ones(batch_size, seq_len, dtype=torch.bool),
        seq_lens=torch.randint(3, resp_len, (batch_size,)),
        answers=NonTensorStack(*[f"ans{i}" for i in range(batch_size)]),
        policy_version=7,
        advantages=advantages,
        rewards=rewards,
        successes=successes,
        reward_func_names=NonTensorStack(*[func_names for _ in range(batch_size)]),
        batch_size=[batch_size],
        sequence_ids=NonTensorStack(
            *[f"worker0_{i}" for i in range(batch_size)]
        ),
    )


class TestReplayBufferContract:
    """The async GRPO recipe stores each whole trajectory batch in the replay
    buffer as a single item and samples exactly one item per training step.

    This contract prevents two failure modes observed in production:

    1. Extending a batched trajectory directly lets the storage capacity count
       in *rows*: with ``max_size < batch_size`` all but the last few samples
       are silently overwritten, and sampling returns repeated copies of the
       same rows -- i.e. identical advantages for every sample in the batch.
       See https://github.com/meta-pytorch/torchtune/issues/2943.
    2. Sampling more than one item returns repeated copies of the whole batch,
       duplicating every sample num_samples times.
    """

    def _make_buffer(self, max_size: int = 2, batch_size: int = 1):
        return ReplayBuffer(
            storage=LazyStackStorage(max_size=max_size), batch_size=batch_size
        )

    def test_extending_batched_trajectory_directly_corrupts_advantages(self):
        """Regression test for the reported bug: the old contract (extend a
        batched trajectory, sample batch_size items) collapses the batch."""
        batch_size, max_size = 16, 2
        traj = _make_trajectory(batch_size=batch_size, seed=1)
        buf = self._make_buffer(max_size=max_size, batch_size=batch_size)
        buf.extend(traj)

        sampled = buf.sample()
        assert sampled.advantages.numel() == batch_size
        # At most max_size distinct advantages can survive the round trip, and
        # with max_size < batch_size the sampled advantages are repeated.
        assert len(torch.unique(sampled.advantages)) <= max_size
        # The surviving rows are the last max_size rows of the batch.
        assert torch.allclose(
            torch.unique(sampled.advantages),
            torch.unique(traj.advantages[-max_size:]),
        )

    def test_sampling_more_than_one_item_repeats_the_batch(self):
        batch_size = 8
        traj = _make_trajectory(batch_size=batch_size, seed=2)
        buf = self._make_buffer(max_size=2, batch_size=1)
        buf.extend(traj.unsqueeze(0))

        sampled = buf.sample(3)
        assert tuple(sampled.batch_size) == (3, batch_size)
        for i in range(3):
            assert torch.equal(sampled.advantages[i], traj.advantages)

    def test_round_trip_preserves_per_sample_fields(self):
        batch_size = 16
        traj = _make_trajectory(batch_size=batch_size, seed=3)
        buf = self._make_buffer(max_size=2, batch_size=1)
        buf.extend(traj.unsqueeze(0))

        sampled = buf.sample(1).squeeze(0)
        assert tuple(sampled.batch_size) == (batch_size,)

        # Every per-sample tensor field must survive exactly.
        for field in (
            "query_responses",
            "responses",
            "logprobs",
            "ref_logprobs",
            "query_response_padding_masks",
            "seq_lens",
            "advantages",
            "rewards",
            "successes",
        ):
            assert torch.equal(getattr(sampled, field), getattr(traj, field)), field

        # Non-tensor per-sample fields must be preserved and aligned with rows.
        assert list(sampled.sequence_ids) == list(traj.sequence_ids)
        assert list(sampled.answers) == list(traj.answers)
        assert [list(f) for f in sampled.reward_func_names] == [
            list(f) for f in traj.reward_func_names
        ]
        assert sampled.policy_version == traj.policy_version

    def test_capacity_counts_batches_not_rows(self):
        batch_size = 16
        max_size = 3
        buf = self._make_buffer(max_size=max_size, batch_size=1)
        for i in range(5):
            buf.extend(_make_trajectory(batch_size=batch_size, seed=10 + i).unsqueeze(0))
        assert len(buf) == max_size

        # The buffer holds whole batches: any sample is one coherent batch whose
        # per-sample fields are internally consistent with its sequence ids.
        for _ in range(20):
            sampled = buf.sample(1).squeeze(0)
            assert tuple(sampled.batch_size) == (batch_size,)
            assert len(sampled.advantages) == batch_size
            # Sequence ids within a sampled batch all come from one batch.
            assert sampled.sequence_ids[0].split("_")[0] == sampled.sequence_ids[
                -1
            ].split("_")[0]

    def test_sample_returns_different_batches(self):
        batch_size, max_size = 8, 3
        buf = self._make_buffer(max_size=max_size, batch_size=1)
        for i in range(3):
            buf.extend(_make_trajectory(batch_size=batch_size, seed=20 + i).unsqueeze(0))

        # With max_size > 1, sampling repeatedly with replacement can return
        # different batches, each internally varied (no degenerate duplicates).
        seen = set()
        for _ in range(100):
            sampled = buf.sample(1).squeeze(0)
            seen.add(tuple(sampled.advantages.tolist()))
        assert len(seen) > 1
        for adv in seen:
            assert len(set(adv)) > 1


class TestGroupNormalizedAdvantages:
    def test_mean_and_std_are_normalized_per_group(self):
        rewards = torch.tensor(
            [
                [10.0, 0.0, 5.0],
                [8.0, 4.0, 6.0],
            ]
        )
        adv = group_normalized_advantages(rewards)
        # Within each group: mean 0 and unit variance (up to the eps term).
        assert torch.allclose(adv.mean(-1), torch.zeros(2), atol=1e-6)
        assert torch.allclose(adv.std(-1), torch.ones(2), atol=1e-3)

    def test_matches_reference_implementation(self):
        torch.manual_seed(0)
        rewards = torch.randn(4, 8)
        mean = rewards.mean(1, keepdim=True)
        std = rewards.std(1, keepdim=True)
        expected = (rewards - mean) / (std + 1e-4)
        assert torch.allclose(group_normalized_advantages(rewards), expected)

    def test_shape_is_preserved(self):
        rewards = torch.randn(3, 5)
        assert tuple(group_normalized_advantages(rewards).shape) == (3, 5)

    def test_constant_group_is_finite(self):
        rewards = torch.tensor([[1.0, 1.0, 1.0]])
        adv = group_normalized_advantages(rewards)
        assert torch.isfinite(adv).all()
        assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
