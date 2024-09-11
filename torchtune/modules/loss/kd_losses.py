# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn.functional as F


class ForwardKLLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        teacher_prob = F.softmax(teacher_logits, dim=-1)
        inf_mask = torch.isinf(student_logits)
        student_logprob = F.log_softmax(student_logits, dim=-1)
        prod_probs = torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (labels != self.ignore_index).int()
        return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


class ForwardKLWithChunkedOutputLoss(torch.nn.Module):
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
        # reshape logits [(bsz, num_tokens/num_chunks, vocab)] -> [(bsz*num_tokens/num_chunks, vocab)]
        teacher_logits = [
            teacher_logits_chunk.reshape(-1, teacher_logits_chunk.size(-1))
            for teacher_logits_chunk in teacher_logits
        ]
        student_logits = [
            student_logits_chunk.reshape(-1, student_logits_chunk.size(-1))
            for student_logits_chunk in student_logits
        ]
        # chunk and reshape labels (bsz, num_tokens, vocab) -> [(bsz*num_tokens/num_chunks, vocab)]
        labels = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]
        total_fkl_loss = 0.0
        for student_chunk, teacher_chunk, label_chunk in zip(
            student_logits, teacher_logits, labels
        ):
            total_fkl_loss += self.fkl_loss(student_chunk, teacher_chunk, label_chunk)
            # teacher_prob = torch.nn.functional.softmax(teacher_chunk, dim=-1)
            # inf_mask = torch.isinf(student_chunk)
            # student_logprob = torch.nn.functional.log_softmax(student_chunk, dim=-1)
            # prod_probs = torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0)
            # x = torch.sum(prod_probs, dim=-1).view(-1)
            # mask = (label_chunk != -100).int()
            # total_fkl_loss += -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return total_fkl_loss / self.num_output_chunks
