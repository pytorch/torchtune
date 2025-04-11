# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch.nn as nn
from torch.distributed.tensor import Partial, Replicate, Shard

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.parallel.style import ParallelStyle
from torchtune.modules.moe._parallelism import (
    ExpertTensorParallel,
    NoParallel,
    PrepareModuleInputOutput,
)
from torchtune.modules.moe.moe import MoE


def decoder_only_tp_training_plan(model: nn.Module) -> Dict[str, ParallelStyle]:
    """
    Helper function to get the tensor parallel plan for Llama4 where only the decoder is parallelized.
    For our implementation to work with fused optimizers, we need to parallelize RMSNorm layers with SequenceParallel.
    However, these layers are incompatible with autoregressive generation (as discussed below), so we
    define a separate parallel plan specifically for training.

    Args:
        model (nn.Module): Model to generate plan for

    Returns:
        Dict[str, Any]: The tensor parallel plan for Llama4 model.
    """
    plan = {
        "decoder": PrepareModuleInput(
            input_kwarg_layouts={
                "tokens": None,
                "mask": None,
                "input_pos": None,
                "input_embeds": Replicate(),
            },
            desired_input_kwarg_layouts={
                "tokens": None,
                "mask": None,
                "input_pos": None,
                "input_embeds": Shard(1),
            },
            use_local_output=True,
        ),
        "decoder.norm": SequenceParallel(),
        "decoder.output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate(),
        ),
    }

    for layer_id, transformer_block in enumerate(model.decoder.layers):

        layer_plan = {
            f"decoder.layers.{layer_id}.sa_norm": SequenceParallel(),
            f"decoder.layers.{layer_id}.attn": PrepareModuleInput(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            f"decoder.layers.{layer_id}.attn.q_proj": ColwiseParallel(),
            f"decoder.layers.{layer_id}.attn.k_proj": ColwiseParallel(),
            f"decoder.layers.{layer_id}.attn.v_proj": ColwiseParallel(),
            f"decoder.layers.{layer_id}.attn.output_proj": RowwiseParallel(
                output_layouts=Shard(1)
            ),
            f"decoder.layers.{layer_id}.mlp_norm": SequenceParallel(),
        }
        if isinstance(transformer_block.mlp, MoE):
            mlp_plan = {
                # input / output sharding on the seqlen dim
                # all-gather for input, reduce-scatter for output
                f"decoder.layers.{layer_id}.mlp": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=True,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Shard(1),),
                ),
                # replicate computation for the router
                f"decoder.layers.{layer_id}.mlp.router.gate": NoParallel(),
                # input Replicate, output Partial
                f"decoder.layers.{layer_id}.mlp.experts": ExpertTensorParallel(),
            }
        else:
            mlp_plan = {
                f"decoder.layers.{layer_id}.mlp": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                f"decoder.layers.{layer_id}.mlp.w1": ColwiseParallel(),
                f"decoder.layers.{layer_id}.mlp.w2": RowwiseParallel(
                    output_layouts=Shard(1)
                ),
                f"decoder.layers.{layer_id}.mlp.w3": ColwiseParallel(),
            }
        layer_plan.update(mlp_plan)
        plan.update(layer_plan)

    return plan


def decoder_only_tp_inference_plan(model: nn.Module) -> Dict[str, ParallelStyle]:
    """
    Helper function to get the tensor parallel plan for Llama4 where only the decoder is parallelized.
    Usage of SequenceParallel requires that tp_dim % seq_len == 0, which will not hold in general
    during autoregressive generation. As a result, we define this plan specifically for usage during generation.
    Args:
        model (nn.Module): Model to generate plan for

    Returns:
        Dict[str, Any]: The tensor parallel plan for Llama4 model.
    """
    plan = {
        "decoder.output": ColwiseParallel(output_layouts=Replicate()),
    }

    for layer_id, transformer_block in enumerate(model.decoder.layers):

        layer_plan = {
            f"decoder.layers.{layer_id}.attn.q_proj": ColwiseParallel(),
            f"decoder.layers.{layer_id}.attn.k_proj": ColwiseParallel(),
            f"decoder.layers.{layer_id}.attn.v_proj": ColwiseParallel(),
            f"decoder.layers.{layer_id}.attn.output_proj": RowwiseParallel(),
        }
        if isinstance(transformer_block.mlp, MoE):
            mlp_plan = {
                # input / output sharding on the seqlen dim
                # all-gather for input, reduce-scatter for output
                f"decoder.layers.{layer_id}.mlp": PrepareModuleOutput(
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Replicate(),),
                ),
                # replicate computation for the router
                f"decoder.layers.{layer_id}.mlp.router.gate": NoParallel(),
                # input Replicate, output Partial
                f"decoder.layers.{layer_id}.mlp.experts": ExpertTensorParallel(),
                f"decoder.layers.{layer_id}.mlp.shared_expert.w1": ColwiseParallel(),
                f"decoder.layers.{layer_id}.mlp.shared_expert.w2": RowwiseParallel(
                    output_layouts=Partial()
                ),
                f"decoder.layers.{layer_id}.mlp.shared_expert.w3": ColwiseParallel(),
            }
        else:
            mlp_plan = {
                f"decoder.layers.{layer_id}.mlp.w1": ColwiseParallel(),
                f"decoder.layers.{layer_id}.mlp.w2": RowwiseParallel(),
                f"decoder.layers.{layer_id}.mlp.w3": ColwiseParallel(),
            }
        layer_plan.update(mlp_plan)
        plan.update(layer_plan)

    return plan


def decoder_only_tp_plan(
    model: nn.Module, inference: bool = False
) -> Dict[str, ParallelStyle]:
    """
    Helper function to get the base tensor parallel plan for Llama3 model, which will also be shared with 3.1, 3.2, and 3.3 models

    Args:
        model (nn.Module): Model to generate plan for (no-op)
        inference (bool): Whether running inference or not.

    Returns:
        Dict[str, Any]: The tensor parallel plan for Llama3 model.
    """
    return (
        decoder_only_tp_inference_plan(model)
        if inference
        else decoder_only_tp_training_plan(model)
    )
