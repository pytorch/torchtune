# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch

from torch.distributed._tensor import Replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.style import ParallelStyle
from torchtune.modules import TransformerSelfAttentionLayer
from torchtune.modules.model_fusion import FusionLayer


# Define the Tensor Parallel plan for Llama3.2 vision model
# TODO: wildcard is not working for the 3.2 deep fusion model. Use `parallelize_llama3_2_vision` for now.
LLAMA3_2_VISION_TP_PLAN = {
    "decoder.tok_embeddings.embedding": RowwiseParallel(
        input_layouts=Replicate(),
    ),
    "decoder.tok_embeddings.fusion_embedding": RowwiseParallel(
        input_layouts=Replicate(),
    ),
    "decoder.output": ColwiseParallel(
        output_layouts=Replicate(),
    ),
    "decoder.layers.*.attn.q_proj": ColwiseParallel(),
    "decoder.layers.*.attn.k_proj": ColwiseParallel(),
    "decoder.layers.*.attn.v_proj": ColwiseParallel(),
    "decoder.layers.*.attn.output_proj": RowwiseParallel(),
    "decoder.layers.*.mlp.w1": ColwiseParallel(),
    "decoder.layers.*.mlp.w2": RowwiseParallel(),
    "decoder.layers.*.mlp.w3": ColwiseParallel(),
    "decoder.layers.*.layer.attn.q_proj": ColwiseParallel(),
    "decoder.layers.*.layer.attn.k_proj": ColwiseParallel(),
    "decoder.layers.*.layer.attn.v_proj": ColwiseParallel(),
    "decoder.layers.*.layer.attn.output_proj": RowwiseParallel(),
    "decoder.layers.*.layer.mlp.w1": ColwiseParallel(),
    "decoder.layers.*.layer.mlp.w2": RowwiseParallel(),
    "decoder.layers.*.layer.mlp.w3": ColwiseParallel(),
    "decoder.layers.*.fusion_layer.attn.q_proj": ColwiseParallel(),
    "decoder.layers.*.fusion_layer.attn.k_proj": ColwiseParallel(),
    "decoder.layers.*.fusion_layer.attn.v_proj": ColwiseParallel(),
    "decoder.layers.*.fusion_layer.attn.output_proj": RowwiseParallel(),
    "decoder.layers.*.fusion_layer.mlp.w1": ColwiseParallel(),
    "decoder.layers.*.fusion_layer.mlp.w2": RowwiseParallel(),
    "decoder.layers.*.fusion_layer.mlp.w3": ColwiseParallel(),
}


def parallelize_llama3_2_vision(
    model: torch.nn.Module,
    tp_device_mesh: DeviceMesh,
) -> None:
    """
    Helper function to parallelize Llama3.2 vision model in a for loop manner.

    Args:
        model (torch.nn.Module): The model to be parallelized.
        tp_device_mesh (DeviceMesh): The device mesh to be used for tensor parallelism.

    Raises:
        ValueError: If the model has a transformer block that is not supported.
    """
    # Parallelize the model
    parallelize_module(
        model,
        tp_device_mesh,
        {
            "decoder.tok_embeddings.embedding": RowwiseParallel(
                input_layouts=Replicate(),
            ),
            "decoder.tok_embeddings.fusion_embedding": RowwiseParallel(
                input_layouts=Replicate(),
            ),
            "decoder.output": ColwiseParallel(
                output_layouts=Replicate(),
            ),
        },
    )

    for transformer_block in model.decoder.layers:
        if isinstance(transformer_block, TransformerSelfAttentionLayer):
            layer_plan = {
                "attn.q_proj": ColwiseParallel(),
                "attn.k_proj": ColwiseParallel(),
                "attn.v_proj": ColwiseParallel(),
                "attn.output_proj": RowwiseParallel(),
                "mlp.w1": ColwiseParallel(),
                "mlp.w2": RowwiseParallel(),
                "mlp.w3": ColwiseParallel(),
            }

            parallelize_module(
                module=transformer_block,
                device_mesh=tp_device_mesh,
                parallelize_plan=layer_plan,
            )
        elif isinstance(transformer_block, FusionLayer):
            layer_plan = {
                "layer.attn.q_proj": ColwiseParallel(),
                "layer.attn.k_proj": ColwiseParallel(),
                "layer.attn.v_proj": ColwiseParallel(),
                "layer.attn.output_proj": RowwiseParallel(),
                "layer.mlp.w1": ColwiseParallel(),
                "layer.mlp.w2": RowwiseParallel(),
                "layer.mlp.w3": ColwiseParallel(),
                "fusion_layer.attn.q_proj": ColwiseParallel(),
                "fusion_layer.attn.k_proj": ColwiseParallel(),
                "fusion_layer.attn.v_proj": ColwiseParallel(),
                "fusion_layer.attn.output_proj": RowwiseParallel(),
                "fusion_layer.mlp.w1": ColwiseParallel(),
                "fusion_layer.mlp.w2": RowwiseParallel(),
                "fusion_layer.mlp.w3": ColwiseParallel(),
            }

            parallelize_module(
                module=transformer_block,
                device_mesh=tp_device_mesh,
                parallelize_plan=layer_plan,
            )
        else:
            raise ValueError("Unsupported transformer block type")


def llama3_2_vision_tp_plan() -> Dict[str, ParallelStyle]:
    """
    Helper function to get the base tensor parallel plan for Llama3.2 vision model.

    Returns:
        Dict[str, Any]: The tensor parallel plan for Llama3.2 vision model.
    """
    return LLAMA3_2_VISION_TP_PLAN
