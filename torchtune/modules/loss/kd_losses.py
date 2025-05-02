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

        sum_masks = torch.sum(mask.view(-1), dim=0)
        if sum_masks == 0:
            return torch.tensor(0.0, device=x.device)
        return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


class ReverseKLLoss(torch.nn.Module):
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

        student_prob = F.softmax(student_logits, dim=-1, dtype=torch.float32)

        student_logprob = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
        teacher_logprob = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)

        inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)

        prod_probs = torch.masked_fill(student_prob * teacher_logprob, inf_mask, 0)
        prod_probs -= torch.masked_fill(student_prob * student_logprob, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)

        mask = (labels != self.ignore_index).int()
        if not normalize:
            return -torch.sum(x * mask.view(-1), dim=0)
        if torch.sum(mask.view(-1), dim=0) == 0:
            return torch.tensor(0.0, device=x.device)
        return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


class SymmetricKLLoss(torch.nn.Module):
    """
    The Symmetric Kullback-Leibler divergence loss for valid indexes.
    Implementation of https://github.com/jongwooko/distillm/blob/17c0f98bc263b1861a02d5df578c84aea652ee65/distillm/losses.py

    Args:
        sym_kd_ratio (float): Ratio of symmetric KL divergence loss.
            When set to 1 this loss reduces to forward KL divergence, when set to 0 this loss reduces to reverse kl divergence.
        ignore_index (int):  Specifies a target value that is ignored and does not contribute to the input gradient.
            The loss is divided over non-ignored targets.
            Default: -100.

    Raises:
        ValueError: If sym_kd_ratio is not in the range [0, 1].
    """

    def __init__(self, sym_kd_ratio: float = 0.5, ignore_index: int = -100):
        super().__init__()
        if sym_kd_ratio < 0.0 or sym_kd_ratio > 1.0:
            raise ValueError("sym_kd_ratio must be in the range [0, 1]")
        self.ignore_index = ignore_index
        self.sym_kd_ratio = sym_kd_ratio
        self.fkl = ForwardKLLoss(ignore_index)
        self.rkl = ReverseKLLoss(ignore_index)

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

        return self.sym_kd_ratio * self.fkl(
            student_logits, teacher_logits, labels, normalize
        ) + (1 - self.sym_kd_ratio) * self.rkl(
            student_logits, teacher_logits, labels, normalize
        )


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

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the fkl_loss function."""
        self.fkl_loss = torch.compile(self.fkl_loss, *args, **kwargs)
        return self

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
            >>> output_chunks = [model.output(chunk) for chunk in h.tensor_split(num_chunks, dim=1)]
            >>> teacher_chunks = [teacher_model.output(chunk) for chunk in h.tensor_split(num_chunks, dim=1)]
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
            for target_chunk in labels.tensor_split(self.num_output_chunks, dim=1)
        ]

        total_fkl_loss = 0.0
        for student_chunk, teacher_chunk, label_chunk in zip(
            student_logits, teacher_logits, labels
        ):
            total_fkl_loss += self.fkl_loss(
                student_chunk, teacher_chunk, label_chunk, normalize=False
            )

        sum_masks = torch.sum(mask.view(-1), dim=0)
        if sum_masks == 0:
            return torch.tensor(0.0, device=student_logits[0].device)

        return total_fkl_loss / torch.sum(mask.view(-1), dim=0)


class ReverseKLWithChunkedOutputLoss(torch.nn.Module):
    """
    Reverse KL with chunked outputs that saves memory by only upcasting one chunk at a time.

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
        self.rkl_loss = ReverseKLLoss(ignore_index)

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the rkl_loss function."""
        self.rkl_loss = torch.compile(self.rkl_loss, *args, **kwargs)
        return self

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
            >>> loss_fn = ReverseKLWithChunkedOutputLoss()
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

        total_rkl_loss = 0.0
        for student_chunk, teacher_chunk, label_chunk in zip(
            student_logits, teacher_logits, labels
        ):
            total_rkl_loss += self.rkl_loss(
                student_chunk, teacher_chunk, label_chunk, normalize=False
            )

        return total_rkl_loss / torch.sum(mask.view(-1), dim=0)


class SymmetricKLWithChunkedOutputLoss(torch.nn.Module):
    """
    Symmetric KL with chunked outputs that saves memory by only upcasting one chunk at a time.

    Since the model is trained with bf16, before computing KL divergence, we have to upcast
    it to fp32 for better accuracy and stability. When upcasting happens, the memory usage doubles.
    Models like llama3 have large vocabulary size and, therefore, have a large output
    result (bsz, num_tokens, vocab_size). If we chunk on the token level, you can still compute
    the cross entropy normally, but upcasting only one chunk at a time saves considerable memory.

    Args:
        num_output_chunks (int): Number of chunks to chunk the output into. Each chunk has shape
            (batch_size, num_tokens / num_output_chunks, vocab_size).
            Default: 8
        sym_kd_ratio (float): Ratio of symmetric KL divergence loss.
            When set to 1 this loss reduces to forward KL divergence, when set to 0 this loss reduces to reverse kl divergence.
            Default: 0.5
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            The loss is divided over non-ignored targets.
            Default: -100
    """

    def __init__(
        self,
        num_output_chunks: int = 8,
        sym_kd_ratio: float = 0.5,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.sym_kd_ratio = sym_kd_ratio
        self.ignore_index = ignore_index
        self.sym_kl_loss = SymmetricKLLoss(
            sym_kd_ratio=self.sym_kd_ratio, ignore_index=self.ignore_index
        )

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the sym_kl_loss function."""
        self.sym_kl_loss = torch.compile(self.sym_kl_loss, *args, **kwargs)
        return self

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
            >>> loss_fn = SymmetricKLWithChunkedOutputLoss()
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

        total_sym_kl_loss = 0.0
        for student_chunk, teacher_chunk, label_chunk in zip(
            student_logits, teacher_logits, labels
        ):
            total_sym_kl_loss += self.sym_kl_loss(
                student_chunk, teacher_chunk, label_chunk, normalize=False
            )

        return total_sym_kl_loss / torch.sum(mask.view(-1), dim=0)
