# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F
from tests.test_utils import assert_expected, fixed_init_model, gpu_test
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import SGD
from torchtune.modules.loss import LigerLinearCrossEntropy
from torchtune.training.seed import set_seed


@gpu_test(gpu_count=1)
class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        return self.output(x)


@pytest.fixture(autouse=True)
def random():
    set_seed(42)


class TestLigerFusedCrossEntropyLoss:
    @pytest.mark.parametrize("compile", [False, True])
    def test_liger_fused_cross_entropy_loss(self, compile):
        """
        Compares LigerFusedCrossEntropyLoss implementation vs standard F.cross_entropy
        """
        # Set up test parameters
        batch_size = 2
        seq_len = 8
        embed_dim = 16
        vocab_size = 100
        ignore_index = -100

        # Create dummy data
        hidden = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)
        targets = torch.randint(
            0,
            vocab_size,
            (
                batch_size,
                seq_len,
            ),
            dtype=torch.long,
        )
        hidden = hidden.cuda()
        targets = targets.cuda()
        # Add some ignored indices
        mask = torch.rand(batch_size, seq_len) < 0.2
        targets[mask] = ignore_index

        # Create a dummy model
        model = Model(vocab_size, embed_dim).cuda()

        # Compute fused CE loss
        loss_fn = LigerLinearCrossEntropy(ignore_index=ignore_index)
        loss_fn.set_model_output(model)
        if compile:
            loss_fn.apply_compile_strategy()
        fused_loss = loss_fn(hidden, targets)

        # Compute standard cross entropy for comparison
        logits = F.linear(
            hidden, model.output.weight, model.output.bias
        )  # [batch_size*seq_len, vocab_size]
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        standard_loss = F.cross_entropy(
            logits, targets, reduction="mean", ignore_index=ignore_index
        )

        # Validate the results are close enough
        assert_expected(fused_loss, standard_loss, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("compile", [False, True])
    def test_liger_fused_cross_entropy_gradients(self, compile):
        """Test gradient flow through full forward/backward pass with optimizer step"""
        # Set up test parameters
        batch_size = 2
        seq_len = 8
        embed_dim = 16
        vocab_size = 100
        ignore_index = -100

        # Create dummy data on GPU
        hidden = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32).cuda()
        targets = torch.randint(
            0,
            vocab_size,
            (
                batch_size,
                seq_len,
            ),
            dtype=torch.long,
        ).cuda()

        # Add ignored indices
        mask = torch.rand(batch_size, seq_len) < 0.2
        targets[mask] = ignore_index

        # Create model with fixed initialization
        model = Model(vocab_size, embed_dim).cuda()
        fixed_init_model(model, min_val=-0.1, max_val=0.1)

        # Store initial weights for comparison
        initial_weight = model.output.weight.detach().clone()
        initial_bias = model.output.bias.detach().clone()

        # Set up loss and optimizer
        loss_fn = LigerLinearCrossEntropy(ignore_index=ignore_index)
        loss_fn.set_model_output(model)
        if compile:
            loss_fn.apply_compile_strategy()
        optimizer = SGD(model.parameters(), lr=0.1)

        # Forward pass
        loss = loss_fn(hidden, targets)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Verify:
        # 1. Gradients were computed
        assert model.output.weight.grad is not None
        assert model.output.bias.grad is not None

        # 2. Parameters actually changed by optimizer
        assert not torch.allclose(model.output.weight, initial_weight)
        assert not torch.allclose(model.output.bias, initial_bias)

        # 3. For DTensor case, verify gradients were scattered back
        if isinstance(model.output.weight, DTensor):
            assert isinstance(model.output.weight.grad, DTensor)
            assert model.output.weight.grad.placements == model.output.weight.placements
