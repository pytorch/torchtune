# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel.style import ParallelStyle


# Define the Tensor Parallel plan for Llama3 model, which will also be shared with 3.1, 3.2, and 3.3 models
BASE_LLAMA_TP_PLAN = {
    "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
    "output": ColwiseParallel(output_layouts=Replicate()),
    "layers.*.attn.q_proj": ColwiseParallel(),
    "layers.*.attn.k_proj": ColwiseParallel(),
    "layers.*.attn.v_proj": ColwiseParallel(),
    "layers.*.attn.output_proj": RowwiseParallel(),
    "layers.*.mlp.w1": ColwiseParallel(),
    "layers.*.mlp.w2": RowwiseParallel(),
    "layers.*.mlp.w3": ColwiseParallel(),
}


def base_llama_tp_plan() -> Dict[str, ParallelStyle]:
    """
    Helper function to get the base tensor parallel plan for Llama3 model, which will also be shared with 3.1, 3.2, and 3.3 models

    Returns:
        Dict[str, Any]: The tensor parallel plan for Llama3 model.
    """
    return BASE_LLAMA_TP_PLAN
