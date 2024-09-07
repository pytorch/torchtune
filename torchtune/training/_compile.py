# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Union

import torch
from torch import nn

from torchtune.modules import (
    TiedEmbeddingTransformerDecoder,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)
from torchtune.modules.loss import CEWithChunkedOutputLoss
from torchtune.utils import get_logger, torch_version_ge

log = get_logger("INFO")


def compile_model(
    model: Union[TransformerDecoder, TiedEmbeddingTransformerDecoder],
    verbose: bool = True,
) -> None:
    """
    Utility to compile a transformer model inplace. On PyTorch nightlies we use per-layer compile
    to reduce compile times. Otherwise we compile the full model, which takes longer.

    Args:
        model (Union[TransformerDecoder, TiedEmbeddingTransformerDecoder]): A transformer model to compile.
        verbose (bool): Whether to log compile info. Default: True
    Returns:
        None

    """
    backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
    if torch_version_ge("2.5.0"):
        if verbose:
            log.info("Compiling model layers with torch.compile...")
        for m in reversed(list(model.modules())):
            if isinstance(m, TransformerSelfAttentionLayer):
                m.compile(backend=backend)
    else:
        if verbose:
            log.info(
                """
                Compiling full model with torch.compile...
                For faster compile times via per-layer compile, please run on PyTorch nightlies.
                """
            )
        model.compile(backend=backend)


def compile_loss(loss: nn.Module, verbose: bool = True) -> None:
    """
    Utility to compile and return loss function. If the loss function is chunked cross-entropy,
    we only compile the upcast + cross-entropy calculation, not the chunking. For other losses
    we compile the entire loss function.

    Args:
        loss (nn.Module): A loss function to compile.
        verbose (bool): Whether to log compile info. Default: True
    Returns:
        loss (nn.Module): loss with either entire module compiled or (in the case of
            CEWithChunkedOutputLoss) only the upcast and cross-entropy calculation compiled.
    """
    backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
    if verbose:
        log.info("Compiling loss with torch.compile...")
    if isinstance(loss, CEWithChunkedOutputLoss):
        loss.compute_cross_entropy = torch.compile(
            loss.compute_cross_entropy, backend=backend
        )
    else:
        loss = torch.compile(loss, backend=backend)
    return loss
