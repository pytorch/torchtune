# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch


class ChunkedCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        num_output_chunks: int = 16,
        ignore_index: int = -100,
        compile_ce: bool = True,
    ):
        super(ChunkedCrossEntropyLoss, self).__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(
            reduction="sum", ignore_index=self.ignore_index
        )
        if compile_ce:
            self.cross_entropy_loss = torch.compile(self.cross_entropy_loss)

    def _compute_cross_entropy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return self.cross_entropy_loss(logits.float(), labels)

    def forward(self, logits: List[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (List[torch.Tensor]): List of chunked logits of length
                ``self.num_output_chunks``, where each chunk has shape
                (batch_size, num_tokens / num_output_chunks, vocab_size).
            labels (torch.Tensor): Ground truth labels of shape (batch_size, num_tokens).

        Returns:
            torch.Tensor: Cross entropy loss of shape (1,).
        """
        total_elements = (labels != self.ignore_index).sum()
        labels = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]
        logits = [
            logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits
        ]

        total_loss = 0.0
        for logits_chunk, labels_chunk in zip(logits, labels):
            total_loss += self.cross_entropy_loss(logits_chunk, labels_chunk)
        return total_loss / total_elements
