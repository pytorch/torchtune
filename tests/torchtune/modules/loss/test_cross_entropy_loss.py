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
from torchtune.modules.loss import LinearCrossEntropyLoss
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


class TestCEWithLinearChunkedOutputLoss:
    def test_linear_chunked_cross_entropy_loss(self):
        """
        Compares torchtune implementation vs standard F.cross_entropy
        """

        # Set up test parameters
        batch_size = 2
        seq_len = 8
        embed_dim = 16
        vocab_size = 100
        ignore_index = -100
        num_chunks = 8

        # Create dummy data
        hidden = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

        # Add some ignored indices
        mask = torch.rand(batch_size, seq_len) < 0.2
        targets[mask] = ignore_index

        # Create a dummy linear layer weight
        model = Model(vocab_size, embed_dim)

        # compute chunked CE
        chunked_ce_loss_fn = LinearCrossEntropyLoss(
            num_output_chunks=num_chunks, ignore_index=ignore_index
        )
        chunked_ce_loss_fn.set_model_output(model)
        chunked_loss = chunked_ce_loss_fn(hidden, targets)

        # Compute standard cross entropy for comparison
        logits = F.linear(
            hidden, model.output.weight
        )  # [batch_size, seq_len, vocab_size]
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        standard_loss = F.cross_entropy(
            logits, targets, reduction="mean", ignore_index=ignore_index
        )

        assert_expected(chunked_loss, standard_loss, rtol=1e-2, atol=1e-2)
