# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.parallel.style import ParallelStyle


# Define the Tensor Parallel plan for Llama3 model, which will also be shared with 3.1, 3.2, and 3.3 models
BASE_LLAMA_TP_PLAN = {
    "tok_embeddings": RowwiseParallel(
        input_layouts=Replicate(), output_layouts=Shard(1)
    ),
    "norm": SequenceParallel(),
    "output": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
    "layers.*.attn": PrepareModuleInput(
        input_layouts=(Shard(1), None),
        desired_input_layouts=(Replicate(), None),
    ),
    "layers.*.mlp": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    "layers.*.sa_norm": SequenceParallel(),
    "layers.*.mlp_norm": SequenceParallel(),
    "layers.*.attn.q_proj": ColwiseParallel(),
    "layers.*.attn.k_proj": ColwiseParallel(),
    "layers.*.attn.v_proj": ColwiseParallel(),
    "layers.*.attn.output_proj": RowwiseParallel(output_layouts=Shard(1)),
    "layers.*.mlp.w1": ColwiseParallel(),
    "layers.*.mlp.w2": RowwiseParallel(output_layouts=Shard(1)),
    "layers.*.mlp.w3": ColwiseParallel(),
}


def base_llama_tp_plan() -> Dict[str, ParallelStyle]:
    """
    Helper function to get the base tensor parallel plan for Llama3 model, which will also be shared with 3.1, 3.2, and 3.3 models

    Returns:
        Dict[str, Any]: The tensor parallel plan for Llama3 model.
    """
    return BASE_LLAMA_TP_PLAN
