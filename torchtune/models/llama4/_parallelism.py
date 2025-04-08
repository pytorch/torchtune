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
    PrepareModuleInputOutput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.parallel.style import ParallelStyle
from torchtune.modules.moe._parallelism import ExpertTensorParallel, NoParallel
from torchtune.modules.moe.moe import MoE


def llama4_decoder_only_tp_plan(model: nn.Module) -> Dict[str, ParallelStyle]:
    """
    Helper function to get the base tensor parallel plan for Llama4, whic

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
                f"decoder.layers.{layer_id}.mlp.shared_expert": PrepareModuleOutput(
                    output_layouts=(Shard(1),),
                    desired_output_layouts=(Replicate(),),
                ),
                f"decoder.layers.{layer_id}.mlp.shared_expert.w1": ColwiseParallel(),
                f"decoder.layers.{layer_id}.mlp.shared_expert.w2": RowwiseParallel(
                    output_layouts=Shard(1),
                ),
                f"decoder.layers.{layer_id}.mlp.shared_expert.w3": ColwiseParallel(),
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
