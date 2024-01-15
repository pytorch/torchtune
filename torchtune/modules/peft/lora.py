# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class LoRALinear(nn.Module):
    """LoRA linear layer as introduced in `LoRA: Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2106.09685>`_.

    LoRA perturbs a given layer via a low-rank approximation where only
    the rank decomposition matrices are trainable. In a linear layer instead of
    :math:`x \\mapsto W_0x` a LoRALinear layer is defined as
    :math:`x \\mapsto W_0x + (\\alpha / r)BAx`, where :math:`r` is the rank of
    the matrices :math:`A` and :math:`B` and :math:`\\alpha` is a scaling factor.
    As in the original implementation, we support dropout before multiplication
    by the low-rank matrices.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        rank (int): rank of the low-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability
        use_bias (bool): whether to include bias in the two low-rank matrices
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.out_dim = out_dim
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=use_bias)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=use_bias)
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/main/loralib/layers.py#L119
        nn.init.zeros_(self.lora_b.weight)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``
        """
        out = self.linear(x)
        lora_out = self.lora_a(self.dropout(x))
        lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)
        return out + lora_out


class LoRAFusedLinear(nn.Module):
    """Class to apply LoRA to subsets of a fused linear layer (for example
    enable LoRA for only Q and V matrices of a fused QKV projection in self-attention).

    This class supports application of LoRA to arbitrary subsets of a fused linear
    layer. See the example given below. For more details on LoRA, see the documentation
    for :func:`~torchtune.modules.peft.LoRALinear`.

    Suppose we have a fused QKV projection mapping input dimension 32 to dimensions
    128, 64, and 64 for Q, K and V respectively (so that the fused projection is
    ``qkv_proj = nn.Linear(32, 256, bias=False)``). If we want to apply LoRA to decompose just the Q and V
    matrices via rank 4 decompositions, we can construct e.g.

    .. code-block:: python

        lora_qv_only = LoRAFusedLinear(
            in_dim=32,
            out_dims=[128, 64, 64],
            apply_lora=[True, False, True],
            rank=4,
            alpha=4.0
        )

    Given an input x, ``lora_qv_only`` will return a matrix of the form

    .. math::

        \\big[(q \\text{_} proj + B_qA_q)x, \\ k \\text{_} proj \\ x, \\ (v \\text{_} proj + B_vA_v)x\\big],

    where :math:`A_q, B_q, A_v, B_v` are the rank decompositions of q and v. The
    embedding dimensions of the above three submatrices are 128, 64, and 64,
    respectively, matching the original embedding dimensions of Q, K, and V.

    Args:
        in_dim (int): input dimension
        out_dims (List[int]): list of output dimensions. The length of this list is
            the number of linear layers that are fused.
        apply_lora (List[bool]): list of which linear layers to which LoRA should be
            applied. E.g. to apply LoRA to Q and V matrices but not K you should
            pass ``[True, False, True]``. Must be the same length as out_dims argument.
        rank (int): rank of each low-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability

    """

    def __init__(
        self,
        in_dim: int,
        out_dims: List[int],
        apply_lora: List[bool],
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(out_dims) == len(
            apply_lora
        ), "Must have same number of output dims and LORA splits"
        self.rank = rank
        self.alpha = alpha
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.all_out_dims_equal = len(set(out_dims)) == 1
        self.out_dim = sum(out_dims)
        self.apply_lora = apply_lora
        self.num_lora_blocks = sum(self.apply_lora)
        self.lora_dims = [
            out_dim for out_dim, lora_split in zip(out_dims, apply_lora) if lora_split
        ]
        self.lora_total_dim = sum(self.lora_dims)
        self.lora_indices = self._get_lora_indices()
        self.linear = nn.Linear(in_features=in_dim, out_features=self.out_dim)
        self.dropout = nn.Dropout(p=dropout)
        if self.num_lora_blocks > 0:
            self.lora_a = nn.Parameter(torch.zeros(rank * self.num_lora_blocks, in_dim))
            self.lora_b = nn.Parameter(torch.zeros(self.lora_total_dim, rank))
            self.reset_lora_parameters()

    def reset_lora_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/main/loralib/layers.py#L119
        nn.init.zeros_(self.lora_b)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def _get_lora_indices(self) -> List[int]:
        """
        This method constructs the indices (along embedding dimension)
        that should have LoRA applied. E.g. if
        ``out_dims = [1, 3, 5, 6]`` and ``apply_lora = [True, False, False, True]``
        then _get_lora_indices will return ``[0, 9, 10, 11, 12, 13, 14]``
        """
        split_indices = [0] + list(np.cumsum(self.out_dims)[:-1])
        lora_indices = []
        for split_size, split_start_idx, use in zip(
            self.out_dims, split_indices, self.apply_lora
        ):
            if use:
                lora_indices.extend(
                    range(split_start_idx, split_start_idx + split_size)
                )
        return lora_indices

    def _parallel_matmul(self, ax: Tensor, b: Tensor) -> Tensor:
        """
        Return parallel ``B @ Ax.T`` along rank dim in ``Ax`` and embed dim in ``B``.
        This can be accomplished using either ``F.linear`` or ``F.conv1d``. We use
        ``F.linear`` in cases where the performance is comparable for clarity, but in
        the case that all output dimensions are equal we use conv1d since it is
        more performant.

        **Detailed discussion on the usage of F.conv1d**

        Suppose we have two tensors X and Y with ``X.shape = (bsz, m, n)`` and
        ``Y.shape = (k, n)``. Then in general

        .. code-block:: python

            (X @ Y.T).transpose(-2, -1) = F.conv1d(X.transpose(-2, -1), Y.unsqueeze(-1)).

        (See e.g. `this blog post <https://sebastianraschka.com/faq/docs/fc-to-conv.html>`_.)
        As applied to a LoRA linear layer we can instead represent

        .. code-block:: python

            B @ (Ax.T) = F.conv1d(Ax.transpose(-2, -1), B.unsqueeze(-1)).transpose(-2, -1),

        with the input x having ``x.shape = (bsz, seq_len, embed_dim)`` and A and B rank
        decomposition matrices such that ``Ax.shape = (bsz, seq_len, rank)`` and
        ``B.shape = (out_dim, rank)``.

        Suppose instead we are calculating ``B @ Ax.T`` in parallel for a fused linear
        layer with LoRA applied to k linear blocks each having dimension d. Then
        ``Ax.shape = (bsz, seq_len, k * rank)`` and ``B.shape = (k * d, rank)``.
        Then our :math:`i^{th}` matmul should be

        .. code-block:: python

            [B @ Ax.T]_i = B[i * d: (i + 1) * d, :] @ Ax[:, :, i * rank : (i + 1) * rank].

        Treating ``Ax`` as a conv1d input and ``B`` as its filter with kernel size 1,
        we see that ``A.transpose(-1, -2)`` has ``k * rank`` as its in_channels ,
        dimension, while ``B.unsqueeze(-1)`` has ``k * out_dim`` as its out_channels
        dimension (see the conv1d docs `here <https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L59-L60>`__).

        Combining this with the usage of the groups parameter (as explained
        `here <https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py#L24-L33>`__),
        it follows that the parallel matmul of ``B @ Ax.T`` over k blocks is given by

        .. code-block:: python

            F.conv1d(Ax.transpose(-1, -2), B.unsqueeze(-1), groups=k).

        """
        # Single block is just usual matmul
        if self.num_lora_blocks == 1:
            return F.linear(ax, b)

        if self.all_out_dims_equal:
            # See detailed explanation above
            return F.conv1d(
                ax.transpose(-1, -2), b.unsqueeze(-1), groups=self.num_lora_blocks
            ).transpose(-1, -2)

        # Otherwise, split Ax along rank dim and B along embed dim
        # We use linear instead of conv1d for clarity, since
        # conv1d doesn't give perf improvements in this case
        ax_splits = ax.chunk(self.num_lora_blocks, dim=-1)
        b_splits = b.split(self.lora_dims)
        return torch.cat(
            [F.linear(ax, b) for ax, b in zip(ax_splits, b_splits)], dim=-1
        )

    def _zero_pad(self, x: Tensor) -> Tensor:
        """
        Given a tensor x of shape ``(..., self.lora_total_dim)``, this method returns a
        tensor of shape ``(..., out_dim)`` filled with the values of x in the indices
        ``self.lora_indices`` and padded with zeros elsewhere. E.g. if

        .. code-block:: python

            x = torch.tensor([
                [1, 2, 3],
                [4, 5, 6]
            ])

        and ``self.lora_indices = [True, True, False, True, False, False, False]``,
        then this method will return

        .. code-block:: python

            torch.tensor([
                [1, 2, 0, 3, 0, 0, 0],
                [4, 5, 0, 6, 0, 0, 0]]
            )

        """
        # If all blocks have LoRA applied, return the tensor as-is (no padding needed)
        if self.num_lora_blocks == len(self.out_dims):
            return x

        # Otherwise create a new tensor of zeros and only fill in LoRA-enabled indices
        x_padded = x.new_zeros(*x.shape[:-1], self.out_dim)
        x_padded = x_padded.index_copy(
            -1, torch.tensor(self.lora_indices, device=x_padded.device), x
        )

        return x_padded

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape ``(bsz, seq_len, in_dim)``

        Returns:
            Tensor: output tensor with shape ``(bsz, seq_len, out_dim)``
        """
        out = self.linear(x)
        # No LoRA blocks -> directly return nn.Linear
        if self.num_lora_blocks == 0:
            return out
        lora_out = self.dropout(x) @ self.lora_a.T
        lora_out = (self.alpha / self.rank) * self._parallel_matmul(
            lora_out, self.lora_b
        )
        lora_out_full = self._zero_pad(lora_out)
        return out + lora_out_full
