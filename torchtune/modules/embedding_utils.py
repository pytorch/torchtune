# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import torch

import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule, fully_shard
from torchtune.modules.model_fusion import (
    DeepFusionModel,
    EarlyFusionModel,
    FusionEmbedding,
)
from torchtune.modules.tied_linear import TiedLinear
from torchtune.utils import get_logger

_log: logging.Logger = get_logger()


def resize_token_embeddings(model: nn.Module, num_embeddings: int) -> None:
    """
    Resizes the token embeddings and the final output projection layer of a ``TransformerDecoder`` model.
    The default init strategy is taking the mean of all the embeddings, new embeddings will be
    instantiated to this value.

    This function modifies the model in-place.

    The primary purpose is to adjust the vocabulary size of a pre-trained model.
    This is useful when fine-tuning a model on a dataset with a different
    vocabulary or when adding special tokens.

    Example:
        >>> model = setup_model(...)
        >>> tokenizer = setup_tokenizer(...)
        >>> resize_token_embedding(model, tokenizer.num_embeddings)

    Args:
        model (nn.Module): The transformer model to modify. The model is
            expected to have ``tok_embeddings`` (an ``nn.Embedding`` layer) and
            ``output`` (e.g., ``nn.Linear`` or ``TiedLinear``) attributes.
        num_embeddings (int): The desired number of embeddings in the resized
            embedding layer and output projection layer.

    Returns:
        None: The function modifies the `model` in-place.

    Raises:
        AssertionError: When trying to resize a model with ``FusionEmbedding``.
    """
    need_resize = _need_to_resize(model, num_embeddings)
    if not need_resize:
        _log.info("Embedding resize not needed, skipping")
        return

    if isinstance(model, DeepFusionModel) or isinstance(model, EarlyFusionModel):
        model = model.decoder
    if isinstance(model.tok_embeddings, FusionEmbedding) and need_resize:
        raise AssertionError("Resize is not supported for FusionEmbedding")

    is_fsdp = isinstance(model, FSDPModule)
    mesh = getattr(model.tok_embeddings.weight, "device_mesh", None)
    is_tied = isinstance(model.output, TiedLinear)

    if is_fsdp:
        model.tok_embeddings.unshard()
        if not is_tied:
            model.output.unshard()

    _resize_token_embeddings_helper(model, num_embeddings)

    if is_fsdp:
        fully_shard(model.tok_embeddings, mesh=mesh)
        if not is_tied:
            fully_shard(model.output, mesh=mesh)


def _need_to_resize(model: nn.Module, num_embeddings: int) -> bool:
    """
    Return True if model's token-embedding matrix size differs from
    ``num_embeddings``, otherwise False.

    Works for either
      • models that expose ``tok_embeddings`` directly, or
      • models that expose it under ``model.decoder.tok_embeddings``.
    """
    emb_layer: Optional[nn.Embedding] = getattr(model, "tok_embeddings", None)
    if emb_layer is None:
        decoder = getattr(model, "decoder", None)
        emb_layer = getattr(decoder, "tok_embeddings", None) if decoder else None

    return emb_layer is not None and emb_layer.num_embeddings != num_embeddings


@torch.no_grad
def _resize_token_embeddings_helper(model: nn.Module, num_embeddings: int) -> None:
    """
    Helper utility that resizes the token embeddings in single device
    """

    def _copy_weights(new_weights: torch.Tensor, old_weights: torch.Tensor) -> None:
        n = min(old_weights.shape[0], new_weights.shape[0])
        new_weights[:n] = old_weights[:n]
        if new_weights.shape[0] > old_weights.shape[0]:
            new_weights[n:] = old_weights[:n].mean(dim=0, keepdim=True)
        new_weights.requires_grad_(old_weights.requires_grad)

    old_embeddings = model.tok_embeddings
    old_num_tokens = model.tok_embeddings.num_embeddings

    _log.info(f"Resizing token embeddings from {old_num_tokens} to {num_embeddings}")
    model.tok_embeddings = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=old_embeddings.embedding_dim,
        padding_idx=old_embeddings.padding_idx,
        device=old_embeddings.weight.device,
        dtype=old_embeddings.weight.dtype,
    )
    _copy_weights(model.tok_embeddings.weight.data, old_embeddings.weight.data)
    del old_embeddings

    output_layer = model.output
    if isinstance(output_layer, TiedLinear):
        model.output.tied_module = model.tok_embeddings
    elif isinstance(output_layer, nn.Linear):
        old_output = model.output
        model.output = nn.Linear(
            in_features=output_layer.in_features,
            out_features=num_embeddings,
            bias=False,
            device=model.tok_embeddings.weight.device,
            dtype=model.tok_embeddings.weight.dtype,
        )
        _copy_weights(model.output.weight.data, old_output.weight.data)
        del old_output
    else:
        _log.warning(
            f"Output layer is not tied and not a recognized Linear layer for resizing. Type: {type(output_layer)}. Skipping."
        )
