# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn.functional as F


class CEWithChunkedOutputLoss(torch.nn.Module):
    """
    Cross-entropy with chunked outputs that saves memory by only upcasting one chunk at a time.

    Whenever the model is trained with bf16, before running CE, we have to upcast
    it to fp32 for better accuracy and stability. When upcasting happens, the memory usage doubles.
    Models like llama3 have large vocabulary size and, therefore, have a large output
    tensor of shape ``(bsz, num_tokens, vocab_size)``. If we chunk on the token level, you can still compute
    the cross entropy normally, but upcasting only one chunk at a time saves considerable memory.

    The CE and upcasting have to be compiled together for better performance.
    When using this class, we recommend using :func:`torch.compile` only on the method ``compute_cross_entropy``.
    The gains from chunking won't be realized if you compile the entire class.

    For more details, please refer to: https://github.com/pytorch/torchtune/pull/1390
    """

    def __init__(self, num_output_chunks: int = 8, ignore_index: int = -100):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index

    def compute_cross_entropy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Upcast logits to fp32 and compute cross entropy loss.
        """
        return F.cross_entropy(
            logits.float(), labels, ignore_index=self.ignore_index, reduction="sum"
        )

    def forward(self, logits: List[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (List[torch.Tensor]): List of chunked logits of length
                ``self.num_output_chunks``, where each chunk has shape
                ``(batch_size, num_tokens / num_output_chunks, vocab_size)``.
            labels (torch.Tensor): Ground truth labels of shape ``(batch_size, num_tokens)``.

        Returns:
            torch.Tensor: Cross entropy loss of shape (1,).

        Example:
            >>> loss_fn = ChunkedCrossEntropyLoss()
            >>>
            >>> h = torch.tensor([bsz, num_tokens, dim])
            >>> output_chunks = [model.output(chunk) for chunk in h.chunk(num_chunks, dim=1)]
            >>>
            >>> labels = torch.tensor([bsz, num_tokens])
            >>> loss = loss_fn(output_chunks, labels)
        """

        total_elements = (labels != self.ignore_index).sum()

        # chunk and reshape labels (bsz, num_tokens, vocab) -> [(bsz*num_tokens/num_chunks, vocab)]
        labels = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]
        # reshape logits [(bsz, num_tokens/num_chunks, vocab)] -> [(bsz*num_tokens/num_chunks, vocab)]
        logits = [
            logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits
        ]

        # compute one chunk at a time
        total_loss = 0.0
        for logits_chunk, labels_chunk in zip(logits, labels):
            total_loss += self.compute_cross_entropy(logits_chunk, labels_chunk)

        return total_loss / total_elements
