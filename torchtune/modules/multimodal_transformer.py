# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn, Tensor

from torchtune.modules import GroupedQueryAttention


class TransformerSelfAttentionLayer(nn.Module):
    """Transformer layer derived from the Llama2 model. Normalization is applied before the attention **and** FF layer.

    Args:
        attn (GroupedQueryAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        attn_norm (Optinoal[nn.Module]): Normalization to be applied before self-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        attn_scale (Optinoal[nn.Module]): Module to scale self-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.
    """

    def __init__(
        self,
        attn: GroupedQueryAttention,
        mlp: nn.Module,
        *,
        attn_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        attn_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.attn_norm = attn_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.attn_scale = attn_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def forward(
        self,
        x: Tensor,
        *,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[Tensor]): Optional boolean tensor which contains the attention mask
                with shape [batch_size x seq_length x seq_length]. This is applied after
                the query-key multiplication and before the softmax. A value of True in row i
                and column j means token i attends to token j. A value of False means token i
                does not attend to token j. If no mask is specified, a causal mask
                is used by default. Default is None.
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.
            kwargs (Dict): transformer layer inputs not relevant to self attention.

        Returns:
            Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        attn_out = self.attn(self.attn_norm(x), mask=mask, input_pos=input_pos)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.attn_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)
        return out


class TransformerCrossAttentionLayer(nn.Module):
    """Cross attention Transformer layer following the same conventions as the TransformerSelfAttentionLayer.
       Normalization is applied before the attention **and** FF layer.

    Args:
        attn (GroupedQueryAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        attn_norm (Optinoal[nn.Module]): Normalization to be applied before self-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        attn_scale (Optinoal[nn.Module]): Module to scale self-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.
    """

    def __init__(
        self,
        attn: GroupedQueryAttention,
        mlp: nn.Module,
        *,
        attn_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        attn_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert (
            attn.pos_embeddings is None
        ), "Positions are not computed for encoder inputs"
        self.attn = attn
        self.mlp = mlp
        self.attn_norm = attn_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.attn_scale = attn_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def _skip_mask(self, mask: Optional[Tensor]) -> Optional[Tensor]:
        """Some tokens in x may not attend to any encoder inputs
        due to the cross attention mask (encoder_mask). This results in
        a full row of the attention matrix being masked out.

        In the example below, the word "the" is masked from every embedding.

        .. code-block:: text

            |emb||emb||emb|
        |The| x    x    x
        |red|      x
        |car| x

        This results in no inputs into the softmax layer which causes a NaN.
        The skip mask removes the output of the attention module and
        mlp resulting in the token being skipped.

        """
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            mask = ~mask
        else:
            mask = torch.isneginf(mask)
        mask = torch.all(mask, dim=-1, keepdim=True)
        return mask

    def forward(
        self,
        x: Tensor,
        *,
        encoder_input: Optional[Tensor] = None,
        encoder_mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            encoder_input (Optional[Tensor]): Optional second input to cross attend with x. Shape
                [batch_size x seq_length x embed_dim] (seq_length and embed_dim may very from x)
            encoder_mask (Optional[Tensor]):  Cross attention boolean tensor with shape
                [batch_size x x_seq_len x encoder_seq_len]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.
            kwargs (Dict): transformer layer inputs not relevant to cross attention.

        Returns:
            Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        # Skip cross attention when no secondary input
        if encoder_input is None:
            return x

        # A mask of tokens (x) with no encoder_input
        skip_mask = self._skip_mask(encoder_mask)

        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        attn_out = self.attn(
            self.attn_norm(x), encoder_input, mask=encoder_mask, input_pos=input_pos
        )
        if skip_mask is not None:
            attn_out.masked_fill_(skip_mask, 0)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.attn_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))
        if skip_mask is not None:
            mlp_out.masked_fill_(skip_mask, 0)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)
        return out
