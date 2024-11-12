# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn.functional as F


class ForwardKLLoss(torch.nn.Module):
    """
    The Kullback-Leibler divergence loss for valid indexes.
    Implementation of https://github.com/jongwooko/distillm/blob/17c0f98bc263b1861a02d5df578c84aea652ee65/distillm/losses.py

    Args:
        ignore_index (int):  Specifies a target value that is ignored and does not contribute to the input gradient.
            The loss is divided over non-ignored targets.
            Default: -100.
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            student_logits (torch.Tensor): logits from student model of shape
                (batch_size*num_tokens, vocab_size).
            teacher_logits (torch.Tensor): logits from teacher model of shape
                (batch_size*num_tokens, vocab_size).
            labels (torch.Tensor): Ground truth labels of shape
                (batch_size, vocab_size).
            normalize (bool): Whether to normalize the loss by the number of unmasked elements.

        Returns:
            torch.Tensor: KL divergence loss of shape (1,).
        """

        teacher_prob = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(student_logits)
        student_logprob = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (labels != self.ignore_index).int()
        if not normalize:
            return -torch.sum(x * mask.view(-1), dim=0)
        if torch.sum(mask.view(-1), dim=0) == 0:
            return torch.tensor(0.0, device=x.device)
        return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


class ForwardKLWithChunkedOutputLoss(torch.nn.Module):
    """
    Forward KL with chunked outputs that saves memory by only upcasting one chunk at a time.

    Since the model is trained with bf16, before computing KL divergence, we have to upcast
    it to fp32 for better accuracy and stability. When upcasting happens, the memory usage doubles.
    Models like llama3 have large vocabulary size and, therefore, have a large output
    result (bsz, num_tokens, vocab_size). If we chunk on the token level, you can still compute
    the cross entropy normally, but upcasting only one chunk at a time saves considerable memory.

    Args:
        num_output_chunks (int): Number of chunks to chunk the output into. Each chunk has shape
            (batch_size, num_tokens / num_output_chunks, vocab_size).
            Default: 8
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            The loss is divided over non-ignored targets.
            Default: -100
    """

    def __init__(self, num_output_chunks: int = 8, ignore_index: int = -100):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.fkl_loss = ForwardKLLoss(ignore_index)

    def forward(
        self,
        student_logits: List[torch.Tensor],
        teacher_logits: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits (List[torch.Tensor]): List of chunked logits from student model of length
                ``self.num_output_chunks``, where each chunk has shape
                (batch_size, num_tokens / num_output_chunks, vocab_size).
            teacher_logits (List[torch.Tensor]): List of chunked logits from teacher model of length
                ``self.num_output_chunks``, where each chunk has shape
                (batch_size, num_tokens / num_output_chunks, vocab_size).
            labels (torch.Tensor): Ground truth labels of shape (batch_size, num_tokens).

        Returns:
            torch.Tensor: KL divergence loss of shape (1,).

        Example:
            >>> loss_fn = ForwardKLWithChunkedOutputLoss()
            >>>
            >>> h = torch.tensor([bsz, num_tokens, dim])
            >>> output_chunks = [model.output(chunk) for chunk in h.chunk(num_chunks, dim=1)]
            >>> teacher_chunks = [teacher_model.output(chunk) for chunk in h.chunk(num_chunks, dim=1)]
            >>> labels = torch.tensor([bsz, num_tokens])
            >>> loss = loss_fn(output_chunks, teacher_chunks, labels)
        """

        # reshape logits [(bsz, num_tokens/num_chunks, vocab)] -> [(bsz*num_tokens/num_chunks, vocab)]
        teacher_logits = [
            teacher_logits_chunk.reshape(-1, teacher_logits_chunk.size(-1))
            for teacher_logits_chunk in teacher_logits
        ]
        student_logits = [
            student_logits_chunk.reshape(-1, student_logits_chunk.size(-1))
            for student_logits_chunk in student_logits
        ]
        mask = (labels != self.ignore_index).int()
        # chunk and reshape labels (bsz, num_tokens, vocab) -> [(bsz*num_tokens/num_chunks, vocab)]
        labels = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]

        total_fkl_loss = 0.0
        for student_chunk, teacher_chunk, label_chunk in zip(
            student_logits, teacher_logits, labels
        ):
            total_fkl_loss += self.fkl_loss(
                student_chunk, teacher_chunk, label_chunk, normalize=False
            )

        return total_fkl_loss / torch.sum(mask.view(-1), dim=0)
