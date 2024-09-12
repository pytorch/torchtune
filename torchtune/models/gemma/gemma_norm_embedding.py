# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class GemmaNormEmbeddings(nn.Embedding):
    """Module with Embedding and normalization specific to Gemma.
    Gemma requires normalization right after the embeddings. By merging both
    steps in a single module, we can utilize directly
    :class:`~torch.modules.TransformerDecoder`.

    For more details about the embedding module, please see
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

    Args:
        num_embeddings (int): size of the dictionary of embeddings.
        embedding_dim (int): the size of each embedding vector.
        *args: Variable length argument list to be passed to the Embedding module.
        **kwargs: Arbitrary keyword arguments to be passed to the Embedding module.

    Example:
        >>> import torch
        >>> from torchtune.models.gemma import GemmaNormEmbeddings
        >>> embeddings = GemmaNormEmbeddings(2, 4)
        >>> x = torch.randint(0, 2, (1, 3)) # ids can be 0 or 1
        >>> print(x)
        >>> print(embeddings(x))
        >>> print(embeddings(x).shape)
        tensor([[1, 0, 0]])
        tensor([[[-0.2152, -2.1914,  2.8491, -0.4824],
                 [-3.6621, -1.0267,  1.5947, -1.7349],
                 [-3.6621, -1.0267,  1.5947, -1.7349]]], grad_fn=<MulBackward0>)
        torch.Size([1, 3, 4])
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, *args, **kwargs):
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return x * torch.tensor(self.embedding_dim**0.5, dtype=x.dtype)
