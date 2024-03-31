# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.modules import TransformerDecoder


def tie_weight(model: TransformerDecoder) -> None:
    """
    Tie the weights of the output embeddings and the token embeddings in the model.

    Args:
        model (TransformerDecoder): The to tie the weights of the output embeddings and the token embeddings.

    Returns:
        None
    """
    output_embeddings = model.output
    input_embeddings = model.tok_embeddings
    output_embeddings.weight = input_embeddings.weight
