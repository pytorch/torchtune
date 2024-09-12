# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected
from torchtune.modules.loss import CEWithChunkedOutputLoss
from torchtune.training.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(42)


class TestCEWithChunkedOutputLoss:
    def test_chunked_cross_entropy_loss(self):
        # Create a sample input and label
        ignore_index = -100
        batch_size = 3
        num_tokens = 50
        vocab_size = 50
        logits = torch.randn(batch_size, num_tokens, vocab_size, dtype=torch.bfloat16)
        labels = torch.randint(
            0, vocab_size, (batch_size, num_tokens), dtype=torch.long
        )

        # add random ignore index to random tokens in the label
        random_indices = torch.randint(0, num_tokens, (batch_size, num_tokens))
        labels[random_indices < num_tokens // 5] = ignore_index

        # chunked CE
        ce_loss = CEWithChunkedOutputLoss(
            num_output_chunks=8, ignore_index=ignore_index
        )
        logits_chunks = logits.chunk(ce_loss.num_output_chunks, dim=1)
        chunked_loss = ce_loss(logits_chunks, labels)

        # vanilla CE
        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
        standard_loss = torch.nn.functional.cross_entropy(
            logits.float(), labels, reduction="mean", ignore_index=ignore_index
        )

        # Assert
        assert_expected(chunked_loss, standard_loss, rtol=1e-2, atol=1e-2)
