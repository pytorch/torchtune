# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torchtune.modules import MultiHeadAttention
from torchtune.modules.attention_utils import _MaskType


class TransformerSelfAttentionLayer(nn.Module):
    """
    Transformer layer derived from the Llama2 model. Normalization is applied before the attention **and** FF layer.

    Args:
        attn (MultiHeadAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm (Optional[nn.Module]): Normalization to be applied before self-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        sa_scale (Optional[nn.Module]): Module to scale self-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.
    """

    def __init__(
        self,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        *,
        sa_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        sa_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.sa_norm = sa_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.sa_scale = sa_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def setup_cache(self, batch_size: int, dtype: torch.dtype) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
        """
        self.attn.setup_cache(batch_size, dtype)

    @property
    def cache_enabled(self) -> bool:
        """Check if the key value caches are setup."""
        return self.attn.kv_cache is not None

    def reset_cache(self):
        """Reset the key value caches."""
        self.attn.reset_cache()

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either a boolean tensor with shape [b x s x s] or a
                :class:`~torch.nn.attention.flex_attention.BlockMask`. If a boolean tensor, a value
                of True in row i and column j means token i attends to token j. A value of False means
                token i does not attend to token j. If no mask is specified, a causal mask
                is used by default. If a :class:`~torch.nn.attention.flex_attention.BlockMask` is passed
                for document masking in a packed sequence via `create_block_mask
                <https://pytorch.org/blog/flexattention/#mask-mods>`_, we use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention.
                Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.
            **kwargs (Dict): transformer layer inputs not relevant to self attention.

        Returns:
            torch.Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        h = self.sa_norm(x)
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.sa_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)
        return out


class TransformerCrossAttentionLayer(nn.Module):
    """
    Cross attention Transformer layer following the same conventions as the TransformerSelfAttentionLayer.
    Normalization is applied before the attention **and** FF layer.

    Args:
        attn (MultiHeadAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        ca_norm (Optional[nn.Module]): Normalization to be applied before cross-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        ca_scale (Optional[nn.Module]): Module to scale cross-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.

    Raises:
        AssertionError: if attn.pos_embeddings is set.
    """

    def __init__(
        self,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        *,
        ca_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        ca_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if attn.pos_embeddings is not None:
            raise AssertionError(
                "Doesn't support positional embeddings for cross attention, \
                because q and k are different sequences."
            )
        self.attn = attn
        self.mlp = mlp
        self.ca_norm = ca_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.ca_scale = ca_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def setup_cache(self, batch_size: int, dtype: torch.dtype) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
        """
        self.attn.setup_cache(batch_size, dtype)

    @property
    def cache_enabled(self) -> bool:
        """Check if the key value caches are setup."""
        return self.attn.kv_cache is not None

    def reset_cache(self):
        """Reset the key value caches."""
        self.attn.reset_cache()

    def _skip_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Some tokens in x may not attend to any encoder inputs
        due to the cross attention mask (encoder_mask). This results in
        a full row of the attention matrix being masked out.

        In the example below, the word "the" is masked from every embedding.
        The False value means a token can't attend to an embedding.

        .. code-block:: text

            |emb||emb||emb|
        |The| F    F    F
        |red| T    F    T
        |car| F    T    T

        This results in no inputs into the softmax layer which causes a NaN.
        The skip mask is used to mask the outputs of attention and
        mlp resulting in the token being skipped.

        The above example would result in a skip mask of: [[True], [False], [False]]
        which specifies which tokens to fully mask out.

        """
        # no skip_mask if no masking
        if mask is None:
            return None
        # negate mask and convert to boolean mask
        if mask.dtype == torch.bool:
            mask = ~mask
        else:
            mask = torch.isneginf(mask)
        # True where all elements in a row are True
        mask = torch.all(mask, dim=-1, keepdim=True)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        *,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape
                [batch_size x token_sequence x embed_dim]
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position i,j means token i can attend
                to embedding j in the decoder. Mask has shape [batch_size x token_sequence x embed_sequence].
                Default is None.
            **kwargs (Dict): transformer layer inputs not relevant to self attention.

        Returns:
            torch.Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        # During decoding, it's possible encoder_input is None because the embeds
        # are already stored in the kv cache.
        empty_cache = not self.cache_enabled or self.attn.kv_cache.size == 0
        # Skip cross attention when no secondary input as it's primary purpose
        # is to attend between x and encoder_input.
        if encoder_input is None and empty_cache:
            return x

        # A mask of tokens (x) with no encoder_input
        skip_mask = self._skip_mask(encoder_mask)

        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        # TODO: Add support for sample packing and bring back input_pos
        attn_out = self.attn(self.ca_norm(x), encoder_input, mask=encoder_mask)
        if skip_mask is not None:
            attn_out.masked_fill_(skip_mask, 0)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.ca_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))
        if skip_mask is not None:
            mlp_out.masked_fill_(skip_mask, 0)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)
        return out


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Return a list of ``n`` identical layers.

    Args:
        module (nn.Module): module to be cloned
        n (int): number of clones

    Returns:
        nn.ModuleList: list of ``n`` identical layers
    """
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder derived from the Llama2 architecture.

    Args:
        tok_embeddings (nn.Embedding): PyTorch embedding layer, to be used to move
            tokens to an embedding space.
        layers (Union[nn.Module, List[nn.Module]]): Transformer Decoder layer or a list of layers.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value. This is used to setup the
            :func:`~torchtune.modules.KVCache`
        head_dim (int): embedding dimension for each head in self-attention. This is used
            to setup the :func:`~torchtune.modules.KVCache`
        norm (nn.Module): Callable that applies normalization to the output of the decoder,
            before final MLP.
        output (Union[nn.Linear, Callable]): Callable that applies a linear transformation to the output of
            the decoder.
        num_layers (Optional[int]): Number of Transformer Decoder layers, only define when
            layers is not a list.
        output_hidden_states (Optional[List[int]]): List of layers (indices) to include in the output

    Raises:
        AssertionError: num_layers is set and layer is a list
        AssertionError: num_layers is not set and layer is an nn.Module

    Note:
        Arg values are checked for correctness (eg: ``attn_dropout`` belongs to [0,1])
        in the module where they are used. This helps reduces the number of raise
        statements in code and improves readability.
    """

    def __init__(
        self,
        *,
        tok_embeddings: nn.Embedding,
        layers: Union[nn.Module, List[nn.Module]],
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        output: Union[nn.Linear, Callable],
        num_layers: Optional[int] = None,
        output_hidden_states: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if num_layers is None:
            if isinstance(layers, nn.Module):
                raise AssertionError(
                    "If num_layers is undefined, it is assumed that a list of layers is provided."
                )
            layers = nn.ModuleList(layers)
        else:
            if not isinstance(layers, nn.Module):
                raise AssertionError("num_layers is defined, layers must be a module")
            layers = _get_clones(layers, num_layers)

        self.tok_embeddings = tok_embeddings
        self.layers = layers
        self.norm = norm
        self.output = output
        self.output_hidden_states = output_hidden_states or []
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal_mask = None
        self.pos = None
        self.num_output_chunks = 0

    def set_num_output_chunks(self, num_output_chunks: int) -> None:
        """Used to save memory in combination with :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss`.
        This should be called before the first forward pass, in the recipe."""
        self.num_output_chunks = num_output_chunks

    def setup_caches(self, batch_size: int, dtype: torch.dtype) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
        """
        for layer in self.layers:
            layer.setup_cache(batch_size, dtype)

        # causal_mask is used during inference to ensure we're attending
        # to the right tokens
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool)
        )
        self.pos = 0

    def caches_are_enabled(self) -> bool:
        """Check if the key value caches are setup."""
        return self.layers[0].cache_enabled

    def reset_caches(self):
        """Reset the key value caches."""
        if not self.caches_are_enabled():
            raise RuntimeError(
                "Key value caches are not setup. Call ``setup_caches()`` first."
            )

        for layer in self.layers:
            layer.reset_cache()

        self.pos = 0

    @torch.compiler.disable
    def chunked_output(self, last_hidden_state: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply output projection in chunks. This should be applied in conjunction with
        :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss` as upcasting to fp32 is done there.

        To use this method, you should first call
        :func:`~torchtune.modules.TransformerDecoder.set_num_output_chunks`.

        Args:
            last_hidden_state (torch.Tensor): last hidden state of the decoder, having shape
                [b, seq_len, embed_dim].

        Returns:
            List[torch.Tensor]: List of num_chunks output tensors, each with shape
                [b, seq_len/num_chunks, out_dim], where out_dim is usually the vocab size.
        """
        return [
            self.output(chunk)
            for chunk in last_hidden_state.chunk(self.num_output_chunks, dim=1)
        ]

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape [b x s]
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either a boolean tensor with shape [b x s x s] or a
                :class:`~torch.nn.attention.flex_attention.BlockMask`. If a boolean tensor, a value
                of True in row i and column j means token i attends to token j. A value of False means
                token i does not attend to token j. If no mask is specified, a causal mask
                is used by default. If a :class:`~torch.nn.attention.flex_attention.BlockMask` is passed
                for document masking in a packed sequence via `create_block_mask
                <https://pytorch.org/blog/flexattention/#mask-mods>`_, we use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention.
                Default is None.
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape [b x s_e x d_e]
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position i,j means token i can attend
                to embedding j in the decoder. Mask has shape [b x s x s_e]. Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: output tensor with shape [b x s x v] or a list of layer
                output tensors defined by ``output_hidden_states`` with the
                final output tensor appended to the list.

        Raises:
            ValueError: if seq_len of x is bigger than max_seq_len
            ValueError: if a mask is provided and the model is in inference mode

        Notation used for tensor shapes:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        if self.causal_mask is not None:
            if mask is not None:
                raise ValueError(
                    "An attention mask was set. Cannot use a non-causal mask for inference"
                )
            # Track the input position
            if input_pos is None:
                input_pos = torch.arange(self.pos, self.pos + seq_len, device=h.device)
            self.pos = input_pos.max() + 1
            # shape: [1, input_pos_len, m_s]
            # in most cases input_pos_len should be 1
            mask = self.causal_mask[None, input_pos]

        hidden = []
        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)
            # shape: [b, s, d]
            h = layer(
                h,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
            )

        # shape: [b, s, d]
        h = self.norm(h)

        if self.num_output_chunks > 0:
            output = self.chunked_output(h)
        else:
            # shape: [b, seq_len, out_dim]
            output = self.output(h).float()

        # Output list if hidden states are requested, otherwise just the output
        # TODO: always output a list to have a consistent output type
        output = output if not hidden else [*hidden, output]
        return output


class TiedEmbeddingTransformerDecoder(nn.Module):
    """
    Transformer Decoder with tied embedding weight. A key difference between
    this class and :class:`~torchtune.modules.TransformerDecoder`
    is that the output projection is replaced with token embeddings weights.

    Args:
        tok_embeddings (nn.Embedding): PyTorch embedding layer, to be used to move
            tokens to an embedding space.
        layers (Union[nn.Module, List[nn.Module]]): Transformer Decoder layer or a list of layers.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value. This is used to setup the
            :func:`~torchtune.modules.KVCache`
        head_dim (int): embedding dimension for each head in self-attention. This is used
            to setup the :func:`~torchtune.modules.KVCache`
        norm (nn.Module): Callable that applies normalization to the output of the decoder,
            before final MLP.
        num_layers (Optional[int]): Number of Transformer Decoder layers, only define when
            layers is not a list.
        output_hidden_states (Optional[List[int]]): List of layers (indices) to include in the output

    Raises:
        AssertionError: num_layers is set and layer is a list
        AssertionError: num_layers is not set and layer is an nn.Module

    Note:
        Arg values are checked for correctness (eg: ``attn_dropout`` belongs to [0,1])
        in the module where they are used. This helps reduces the number of raise
        statements in code and improves readability.
    """

    def __init__(
        self,
        *,
        tok_embeddings: nn.Embedding,
        layers: Union[nn.Module, List[nn.Module]],
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        num_layers: Optional[int] = None,
        output_hidden_states: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if num_layers is None:
            if isinstance(layers, nn.Module):
                raise AssertionError(
                    "If num_layers is undefined, it is assumed that a list of layers is provided."
                )
            layers = nn.ModuleList(layers)
        else:
            if not isinstance(layers, nn.Module):
                raise AssertionError("num_layers is defined, layers must be a module")
            layers = _get_clones(layers, num_layers)

        self.tok_embeddings = tok_embeddings
        self.layers = layers
        self.norm = norm
        self.output_hidden_states = output_hidden_states or []
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal_mask = None
        self.pos = None
        self.num_output_chunks = 0

    def set_num_output_chunks(self, num_output_chunks: int) -> None:
        """Used to save memory in combination with :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss`.
        This should be called before the first forward pass, in the recipe."""
        self.num_output_chunks = num_output_chunks

    def setup_caches(self, batch_size: int, dtype: torch.dtype) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
        """
        for layer in self.layers:
            layer.setup_cache(batch_size, dtype)

        # causal_mask is used during inference to ensure we're attending
        # to the right tokens
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool)
        )
        self.pos = 0

    def caches_are_enabled(self) -> bool:
        """Check if the key value caches are setup."""
        return self.layers[0].cache_enabled

    def reset_caches(self):
        """Reset the key value caches."""
        if not self.caches_are_enabled():
            raise RuntimeError(
                "Key value caches are not setup. Call ``setup_caches()`` first."
            )

        for layer in self.layers:
            layer.reset_cache()

        self.pos = 0

    @torch.compiler.disable
    def chunked_output(self, last_hidden_state: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply output projection in chunks. This should be applied in conjunction with
        :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss` as upcasting to fp32 is done there.

        To use this method, you should first call
        :func:`~torchtune.modules.TiedEmbeddingTransformerDecoder.set_num_output_chunks`.

        Args:
            last_hidden_state (torch.Tensor): last hidden state of the decoder, having shape
                [b, seq_len, embed_dim].

        Returns:
            List[torch.Tensor]: List of num_chunks output tensors, each with shape
                [b, seq_len/num_chunks, out_dim], where out_dim is usually the vocab size.
        """
        return [
            F.linear(chunk, self.tok_embeddings.weight)
            for chunk in last_hidden_state.chunk(self.num_output_chunks, dim=1)
        ]

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape [b x s]
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either a boolean tensor with shape [b x s x s] or a
                :class:`~torch.nn.attention.flex_attention.BlockMask`. If a boolean tensor, a value
                of True in row i and column j means token i attends to token j. A value of False means
                token i does not attend to token j. If no mask is specified, a causal mask
                is used by default. If a :class:`~torch.nn.attention.flex_attention.BlockMask` is passed
                for document masking in a packed sequence via `create_block_mask
                <https://pytorch.org/blog/flexattention/#mask-mods>`_, we use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention.
                Default is None.
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape [b x s_e x d_e]
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position i,j means token i can attend
                to embedding j in the decoder. Mask has shape [b x s x s_e]. Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            torch.Tensor: output tensor with shape [b x s x v] or a list of layer
                output tensors defined by ``output_hidden_states`` with the
                final output tensor appended to the list.

        Raises:
            ValueError: if seq_len of x is bigger than max_seq_len
            ValueError: if a mask is provided and the model is in inference mode

        Notation used for tensor shapes:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        if self.causal_mask is not None:
            if mask is not None:
                raise ValueError(
                    "An attention mask was set. Cannot use a non-causal mask for inference"
                )
            # Track the input position
            if input_pos is None:
                input_pos = torch.arange(self.pos, self.pos + seq_len, device=h.device)
            self.pos = input_pos.max() + 1
            # shape: [1, input_pos_len, m_s]
            # in most cases input_pos_len should be 1
            mask = self.causal_mask[None, input_pos]

        hidden = []
        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)
            # shape: [b, s, d]
            h = layer(
                h,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
            )

        # shape: [b, s, d]
        h = self.norm(h)

        if self.num_output_chunks > 0:
            output = self.chunked_output(h)
        else:
            # shape: [b, seq_len, out_dim]
            output = F.linear(h, self.tok_embeddings.weight).float()

        # Output list if hidden states are requested, otherwise just the output
        # TODO: always output a list to have a consistent output type
        output = output if not hidden else [*hidden, output]
        return output
