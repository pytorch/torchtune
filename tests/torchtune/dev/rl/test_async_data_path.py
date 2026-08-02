# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import pytest
import torch
from torch import nn

from tensordict import NonTensorStack

from torchtune.dev.rl.datatypes.trajectory import Trajectory
from torchtune.dev.rl.linear_grpo_loss import LinearGRPOLoss
from torchtune.dev.rl.rewards import group_normalized_advantages
from torchrl.data import LazyStackStorage, ReplayBuffer, RoundRobinWriter


def _make_trajectory(batch_size, seq_len, resp_len, num_funcs, seed=0, advantages=None):
    """Build a trajectory in the same layout PostProcessingWorker emits."""
    torch.manual_seed(seed)
    rewards = torch.randn(batch_size, num_funcs)
    successes = torch.rand(batch_size, num_funcs) > 0.5
    func_names = [f"reward_fn_{i}" for i in range(num_funcs)]
    if advantages is None:
        advantages = torch.randn(batch_size)
    return Trajectory(
        query_responses=torch.randint(0, 100, (batch_size, seq_len)),
        responses=torch.randint(0, 100, (batch_size, resp_len)),
        logprobs=torch.randn(batch_size, resp_len),
        ref_logprobs=torch.randn(batch_size, resp_len),
        query_response_padding_masks=torch.ones(batch_size, seq_len, dtype=torch.bool),
        seq_lens=torch.randint(1, resp_len, (batch_size,)),
        answers=NonTensorStack(*[f"ans{i}" for i in range(batch_size)]),
        policy_version=3,
        advantages=advantages,
        rewards=rewards,
        successes=successes,
        reward_func_names=NonTensorStack(*[func_names for _ in range(batch_size)]),
        batch_size=[batch_size],
        sequence_ids=NonTensorStack(
            *[f"worker0_{i}" for i in range(batch_size)]
        ),
    )


class TestAdvantageGradientOracle:
    """Advantages must flow into the optimizer proportionally to their value.

    The GRPO policy loss with ``kl_coeff=0`` reduces to ``-advantages.mean()``
    per batch, so the gradient with respect to any model parameter is exactly
    linear in the per-sample advantages. These tests pin that down so that any
    regression that collapses advantages (e.g. all samples seeing the same
    value, the #2943 symptom) is caught by construction.
    """

    def _setup(self, batch_size=4, seq_len=6):
        torch.manual_seed(0)
        num_output_chunks = 1
        loss_fn = LinearGRPOLoss(
            num_output_chunks=num_output_chunks, kl_coeff=0.0
        )
        head = nn.Linear(8, 32)
        loss_fn.linear_projection = head
        hidden = torch.randn(batch_size, seq_len, 8)
        targets = torch.randint(0, 32, (batch_size, seq_len))
        ref_logprobs = torch.randn(batch_size, seq_len)
        masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        return loss_fn, hidden, targets, ref_logprobs, masks

    def test_loss_equals_neg_advantage_mean(self):
        loss_fn, hidden, targets, ref_logprobs, masks = self._setup()
        advantages = torch.tensor([1.0, -2.0, 0.5, 0.25])
        loss, policy_loss, *_ = loss_fn(
            hidden, targets, ref_logprobs, advantages, masks
        )
        # kl_coeff=0 -> the policy term is exactly -advantages[:, None] per token
        assert torch.allclose(loss, -advantages.mean(), atol=1e-6)
        assert torch.allclose(policy_loss, advantages.mean(), atol=1e-6)

    def test_gradient_scales_linearly_with_advantages(self):
        loss_fn, hidden, targets, ref_logprobs, masks = self._setup()

        def grad_norm(advantages):
            loss, *_ = loss_fn(hidden, targets, ref_logprobs, advantages, masks)
            loss.backward(retain_graph=True)
            norm = torch.cat(
                [p.grad.flatten() for p in loss_fn.parameters() if p.grad is not None]
            )
            loss_fn.zero_grad()
            return norm

        adv = torch.tensor([1.0, -2.0, 0.5, 0.25])
        g1 = grad_norm(adv)
        g2 = grad_norm(2.0 * adv)
        assert torch.allclose(g2, 2.0 * g1, atol=1e-6)

    def test_gradient_is_additive_over_samples(self):
        """Per-sample advantages add: grad([2,1]) == 2*grad([1,0]) + grad([0,1])."""
        loss_fn, hidden, targets, ref_logprobs, masks = self._setup(batch_size=2)
        ones_hot = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]

        def grad_vec(advantages):
            loss, *_ = loss_fn(hidden, targets, ref_logprobs, advantages, masks)
            loss.backward(retain_graph=True)
            vec = torch.cat(
                [p.grad.flatten() for p in loss_fn.parameters() if p.grad is not None]
            )
            loss_fn.zero_grad()
            return vec

        g_a, g_b = (grad_vec(v) for v in ones_hot)
        combined = grad_vec(torch.tensor([2.0, 1.0]))
        assert torch.allclose(combined, 2.0 * g_a + g_b, atol=1e-6)


class TestAsyncDataPathFuzz:
    """Fuzz the postprocessing -> replay buffer -> sampling round trip."""

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize(
        "batch_size,group_size,resp_len,num_funcs,capacity",
        [
            (2, 2, 8, 1, 2),  # config-like: buffer capacity in batches
            (4, 2, 16, 2, 3),
            (1, 4, 32, 3, 1),  # capacity 1: only the last batch survives
            (2, 3, 11, 4, 5),
        ],
    )
    def test_round_trip_preserves_everything(
        self, seed, batch_size, group_size, resp_len, num_funcs, capacity
    ):
        grpo_samples = batch_size * group_size
        buffer = ReplayBuffer(
            storage=LazyStackStorage(max_size=capacity),
            batch_size=1,
            writer=RoundRobinWriter(),
        )
        batch = _make_trajectory(
            grpo_samples, resp_len + 8, resp_len, num_funcs, seed=seed
        )

        # Storage counts each whole batch as one item (postprocessing contract).
        buffer.extend(batch.unsqueeze(0))
        sampled = buffer.sample(1).squeeze(0)

        assert tuple(sampled.batch_size) == (grpo_samples,)
        # The only batch in the buffer is returned wholesale: per-sample fields
        # are bit-for-bit the ones that went in.
        assert torch.equal(sampled.advantages, batch.advantages)
        assert torch.equal(sampled.rewards, batch.rewards)
        assert torch.equal(sampled.successes, batch.successes)
        assert sampled.reward_func_names == batch.reward_func_names
        assert sampled.sequence_ids == batch.sequence_ids
        assert sampled.policy_version == batch.policy_version
        assert torch.equal(sampled.logprobs, batch.logprobs)

    @pytest.mark.parametrize("seed", range(5))
    def test_advantages_vary_within_groups_after_round_trip(
        self, seed, group_size=4, batch_size=3, resp_len=10, num_funcs=2
    ):
        """Regression for #2943: sampled advantages must NOT be identical."""
        grpo_samples = batch_size * group_size
        rewards = torch.randn(grpo_samples, num_funcs)
        advantages = group_normalized_advantages(
            rewards.reshape(batch_size, group_size, -1).sum(-1)
        ).reshape(-1)

        buffer = ReplayBuffer(
            storage=LazyStackStorage(max_size=2),
            batch_size=1,
            writer=RoundRobinWriter(),
        )
        traj = _make_trajectory(grpo_samples, resp_len + 8, resp_len, num_funcs, seed)
        traj = Trajectory(
            query_responses=traj.query_responses,
            responses=traj.responses,
            logprobs=traj.logprobs,
            ref_logprobs=traj.ref_logprobs,
            query_response_padding_masks=traj.query_response_padding_masks,
            seq_lens=traj.seq_lens,
            answers=traj.answers,
            policy_version=traj.policy_version,
            advantages=advantages,
            rewards=rewards,
            successes=traj.successes,
            reward_func_names=traj.reward_func_names,
            batch_size=grpo_samples,
            sequence_ids=traj.sequence_ids,
        )
        buffer.extend(traj.unsqueeze(0))
        sampled = buffer.sample(1).squeeze(0)

        adv = sampled.advantages.reshape(batch_size, group_size)
        # Every group is centered at zero (GRPO normalization survived the trip).
        assert torch.allclose(adv.mean(-1), torch.zeros(batch_size), atol=1e-5)
        # The per-sample advantages are distinct within a group (the #2943 bug
        # produced a constant value across all samples instead).
        assert adv.std(-1).gt(1e-6).all()

    def test_capacity_counts_batches_across_multiple_writes(self):
        buffer = ReplayBuffer(
            storage=LazyStackStorage(max_size=2),
            batch_size=1,
            writer=RoundRobinWriter(),
        )
        first = _make_trajectory(4, 16, 8, 1, seed=1)
        second = _make_trajectory(4, 16, 8, 1, seed=2)
        third = _make_trajectory(4, 16, 8, 1, seed=3)
        buffer.extend(first.unsqueeze(0))
        buffer.extend(second.unsqueeze(0))
        buffer.extend(third.unsqueeze(0))

        # Capacity 2 batches: the first batch was evicted, the last two remain,
        # and sampling always returns one coherent whole batch.
        for _ in range(10):
            sampled = buffer.sample(1).squeeze(0)
            assert torch.equal(
                sampled.advantages, second.advantages
            ) or torch.equal(sampled.advantages, third.advantages)


class TestRewardFuncNamesLabeling:
    def test_per_sample_names_from_reward_outputs(self):
        """The collector emits trajectories without reward labels; the
        postprocessing worker labels them from the reward functions it ran."""
        from torch import tensor

        from torchtune.dev.rl.rewards import RewardOutput
        from torchtune.dev.rl.workers.postprocessing import (
            reward_func_names_per_sample,
        )

        reward_outputs = [
            RewardOutput(
                reward_base_name="math_correctness",
                total_reward=tensor([1.0, 0.0]),
                successes=tensor([1, 0]),
            ),
            RewardOutput(
                reward_base_name="formatting",
                total_reward=tensor([1.0, 1.0]),
                successes=tensor([1, 1]),
            ),
        ]
        names = reward_func_names_per_sample(reward_outputs, num_samples=3)
        assert len(names) == 3
        for i in range(3):
            assert names[i].data == ["math_correctness", "formatting"]

    def test_labels_survive_buffer_round_trip(self):
        """Per-sample reward names survive sampling from the replay buffer."""
        buffer = ReplayBuffer(
            storage=LazyStackStorage(max_size=2),
            batch_size=1,
            writer=RoundRobinWriter(),
        )
        traj = _make_trajectory(4, 16, 8, 2, seed=0)
        buffer.extend(traj.unsqueeze(0))
        sampled = buffer.sample(1).squeeze(0)
        assert sampled.reward_func_names == traj.reward_func_names


class TestGroupAdvantageAggregation:
    def test_aggregation_matches_reference(self):
        """Reproduce the exact aggregation used by the async pipeline.

        The reference is computed from first principles in float64, which is
        independent of the implementation's eps/mean/std choices.
        """
        torch.manual_seed(7)
        for batch_size, group_size, num_funcs in itertools.product(
            (1, 2, 4), (2, 4, 8), (1, 3)
        ):
            rewards = torch.randn(batch_size, group_size, num_funcs)
            r = rewards.sum(-1).double()
            expected = (r - r.mean(1, keepdim=True)) / (
                r.std(1, keepdim=True) + 1e-4
            )
            got = group_normalized_advantages(rewards.sum(-1))
            assert torch.allclose(got, expected.float(), atol=1e-6)
            assert torch.allclose(got.mean(-1), torch.zeros(batch_size), atol=1e-6)
