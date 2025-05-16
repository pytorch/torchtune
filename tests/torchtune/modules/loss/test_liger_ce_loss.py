# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F
from tests.test_utils import assert_expected
from torch import nn
from torchtune.modules.loss import LigerFusedCrossEntropyLoss
from torchtune.training.seed import set_seed


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
    def test_liger_fused_cross_entropy_loss(self):
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
        hidden = torch.randn(batch_size * seq_len, embed_dim, dtype=torch.float32)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,), dtype=torch.long)
        hidden = hidden.cuda()
        targets = targets.cuda()
        # Add some ignored indices
        mask = torch.rand(batch_size * seq_len) < 0.2
        targets[mask] = ignore_index

        # Create a dummy model
        model = Model(vocab_size, embed_dim).cuda()

        # Compute fused CE loss
        fused_ce_loss_fn = LigerFusedCrossEntropyLoss(ignore_index=ignore_index)
        fused_ce_loss_fn.set_model_output(model)
        fused_loss = fused_ce_loss_fn(hidden, targets)

        # Compute standard cross entropy for comparison
        logits = F.linear(hidden, model.output.weight, model.output.bias)  # [batch_size*seq_len, vocab_size]
        standard_loss = F.cross_entropy(
            logits, targets, reduction="sum", ignore_index=ignore_index
        )

        # Validate the results are close enough
        assert_expected(fused_loss, standard_loss, rtol=1e-2, atol=1e-2)

    def test_liger_fused_cross_entropy_loss_with_reshape(self):
        """
        Tests LigerFusedCrossEntropyLoss with batch x seq_len input that needs reshaping
        """
        # Set up test parameters
        batch_size = 3
        seq_len = 6
        embed_dim = 32
        vocab_size = 50
        ignore_index = -100

        # Create dummy data with batch x seq_len dimensions
        hidden = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        hidden = hidden.cuda()
        targets = targets.cuda()
        # Add some ignored indices
        mask = torch.rand(batch_size, seq_len) < 0.2
        targets[mask] = ignore_index

        # Create a dummy model
        model = Model(vocab_size, embed_dim).cuda()

        # Reshape to match expected input shape for fused loss
        hidden_reshaped = hidden.reshape(-1, embed_dim)
        targets_reshaped = targets.reshape(-1)

        # Compute fused CE loss
        fused_ce_loss_fn = LigerFusedCrossEntropyLoss(ignore_index=ignore_index)
        fused_ce_loss_fn.set_model_output(model)
        fused_loss = fused_ce_loss_fn(hidden_reshaped, targets_reshaped)

        # Compute standard cross entropy for comparison
        logits = model(hidden_reshaped)  # [batch_size*seq_len, vocab_size]
        standard_loss = F.cross_entropy(
            logits, targets_reshaped, reduction="sum", ignore_index=ignore_index
        )

        # Validate the results are close enough
        assert_expected(fused_loss, standard_loss, rtol=1e-2, atol=1e-2)

