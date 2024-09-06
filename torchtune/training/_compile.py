# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Union

from torchtune.modules import (
    CEWithChunkedOutputLoss,
    TiedEmbeddingTransformerDecoder,
    TransformerDecoder,
)
from torchtune.utils import get_logger, torch_version_ge

log = get_logger("INFO")


def compile_model(
    model: Union[TransformerDecoder, TiedEmbeddingTransformerDecoder]
) -> None:
    """
    Utility to compile a transformer model inplace. On PyTorch nightlies we use per-layer compile
    to reduce compile times. Otherwise we compile the full model, which takes longer.

    Args:
        model (Union[TransformerDecoder, TiedEmbeddingTransformerDecoder]): A transformer model to compile.
    Returns:
        None
    """
    backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
    if torch_version_ge("2.5.0"):
        log.info("Compiling model layers with torch.compile...")
        for m in reversed(list(model.modules())):
            if isinstance(m, modules.transformer.TransformerSelfAttentionLayer):
                m.compile(backend=backend)
    else:
        log.info(
            """
        Compiling full model with torch.compile...
        For faster compile times via per-layer compile, please run on PyTorch nightlies.
        """
        )


def compile_loss(loss: nn.Module) -> None:
    """
    Utility to compile loss function inplace. If the loss function is chunked cross-entropy,
    we only compile the upcast + cross-entropy calculation, not the chunking. For other losses
    we compile the entire loss function.

    Args:
        loss (nn.Module): A loss function to compile.
    Returns:
        None
    """
    backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
    if isinstance(loss, CEWithChunkedOutputLoss):
        loss.compute_cross_entropy.compile(backend=backend)
    else:
        loss.compile(backend=backend)
