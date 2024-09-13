# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected
from torchtune.modules.loss import ForwardKLLoss, ForwardKLWithChunkedOutputLoss
from torchtune.training.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(42)


class TestForwardKLWithChunkedOutputLoss:
    def test_forward_kl_loss(self):
        # Create a sample input and label
        ignore_index = -100
        batch_size = 3
        num_tokens = 50
        vocab_size = 50
        logits = torch.randn(batch_size, num_tokens, vocab_size, dtype=torch.bfloat16)
        teacher_logits = torch.randn(
            batch_size, num_tokens, vocab_size, dtype=torch.bfloat16
        )
        labels = torch.randint(
            0, vocab_size, (batch_size, num_tokens), dtype=torch.long
        )

        # add random ignore index to random tokens in the label
        random_indices = torch.randint(0, num_tokens, (batch_size, num_tokens))
        labels[random_indices < num_tokens // 5] = ignore_index

        # chunked FKL
        chunked_fkl_loss = ForwardKLWithChunkedOutputLoss(
            num_output_chunks=8, ignore_index=ignore_index
        )
        logits_chunks = logits.chunk(chunked_fkl_loss.num_output_chunks, dim=1)
        teacher_logits_chunks = teacher_logits.chunk(
            chunked_fkl_loss.num_output_chunks, dim=1
        )
        chunked_loss = chunked_fkl_loss(logits_chunks, teacher_logits_chunks, labels)

        # vanilla FKL
        fkl_loss = ForwardKLLoss(ignore_index=ignore_index)
        logits = logits.reshape(-1, logits.size(-1))
        teacher_logits = teacher_logits.reshape(-1, teacher_logits.size(-1))
        labels = labels.reshape(-1)
        standard_loss = fkl_loss(logits, teacher_logits, labels)

        # Assert
        assert_expected(chunked_loss, standard_loss, rtol=1e-2, atol=1e-2)

    def test_forward_kl_loss_expected(self):
        student_logits = torch.tensor(
            [
                [
                    [1.1250, -0.4102, -0.0879, -2.5000],
                    [0.2676, 0.3535, 0.8711, -1.4688],
                    [-0.1084, 1.6641, 0.0084, 0.1196],
                    [0.5000, -0.6406, -0.2236, -1.5938],
                ],
                [
                    [-1.5312, -1.9219, 0.0000, -0.5039],
                    [-1.5391, 1.5312, 0.5820, 0.2695],
                    [-0.3887, 1.2188, 0.0000, 0.6055],
                    [0.5000, 1.3828, 0.1309, -1.0312],
                ],
            ],
            dtype=torch.bfloat16,
        )
        teacher_logits = torch.tensor(
            [
                [
                    [-0.0381, -1.2578, -1.2031, 0.0947],
                    [-0.7852, 0.4492, 1.5547, 0.0972],
                    [0.8203, 0.0012, 0.7656, 0.3477],
                    [-1.5781, 0.4297, 0.5977, 0.3926],
                ],
                [
                    [1.5156, 0.1641, 2.0781, -0.7734],
                    [-0.5898, 0.4453, -0.7969, 0.6328],
                    [0.6289, -0.8359, 0.9258, 0.2109],
                    [0.0006, 0.5195, 3.2344, -1.5781],
                ],
            ],
            dtype=torch.bfloat16,
        )
        labels = torch.tensor([[0, 3, 3, 1], [1, 1, 1, 1]])
        expected_loss = torch.tensor(1.7209, dtype=torch.float32)

        # chunked FKL loss
        chunked_fkl_loss = ForwardKLWithChunkedOutputLoss(
            num_output_chunks=2, ignore_index=-100
        )
        student_logits_chunks = student_logits.chunk(
            chunked_fkl_loss.num_output_chunks, dim=1
        )
        teacher_logits_chunks = teacher_logits.chunk(
            chunked_fkl_loss.num_output_chunks, dim=1
        )
        chunked_loss = chunked_fkl_loss(
            student_logits_chunks, teacher_logits_chunks, labels
        )

        # vanilla FKL loss
        fkl_loss = ForwardKLLoss(ignore_index=-100)
        standard_loss = fkl_loss(student_logits, teacher_logits, labels)

        # assert
        assert_expected(chunked_loss, expected_loss, rtol=1e-2, atol=1e-2)
        assert_expected(standard_loss, expected_loss, rtol=1e-2, atol=1e-2)
