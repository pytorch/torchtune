# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel.style import ParallelStyle


# Define the Tensor Parallel plan for Llama3.2 vision model
LLAMA3_2_VISION_TP_PLAN = {
    "decoder.tok_embeddings.embedding": RowwiseParallel(
        input_layouts=Replicate()
    ),
    "decoder.tok_embeddings.fusion_embedding": RowwiseParallel(
        input_layouts=Replicate()
    ),
    "decoder.output": ColwiseParallel(
        output_layouts=Replicate()
    ),
    "decoder.layers.*.layer.attn.q_proj": ColwiseParallel(),
    "decoder.layers.*.layer.attn.k_proj": ColwiseParallel(),
    "decoder.layers.*.layer.attn.v_proj": ColwiseParallel(),
    "decoder.layers.*.layer.attn.output_proj": RowwiseParallel(),
    "decoder.layers.*.layer.mlp.w1": ColwiseParallel(),
    "decoder.layers.*.layer.mlp.w2": RowwiseParallel(),
    "decoder.layers.*.layer.mlp.w3": ColwiseParallel(),
    # "decoder.layers.*.fusion_layer.attn.q_proj": ColwiseParallel(),
    # "decoder.layers.*.fusion_layer.attn.k_proj": ColwiseParallel(),
    # "decoder.layers.*.fusion_layer.attn.v_proj": ColwiseParallel(),
    # "decoder.layers.*.fusion_layer.attn.output_proj": RowwiseParallel(),
    # "decoder.layers.*.fusion_layer.mlp.w1": ColwiseParallel(),
    # "decoder.layers.*.fusion_layer.mlp.w2": RowwiseParallel(),
    # "decoder.layers.*.fusion_layer.mlp.w3": ColwiseParallel(),
}


def llama3_2_vision_tp_plan() -> Dict[str, ParallelStyle]:
    """
    Helper function to get the base tensor parallel plan for Llama3.2 vision model.

    Returns:
        Dict[str, Any]: The tensor parallel plan for Llama3.2 vision model.
    """
    return LLAMA3_2_VISION_TP_PLAN
