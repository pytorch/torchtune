# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel.style import ParallelStyle

from torchao.float8.float8_tensor_parallel import (
    Float8ColwiseParallel,
    Float8RowwiseParallel,
)


def base_llama_tp_plan(enable_float8: bool) -> Dict[str, ParallelStyle]:
    """
    Helper function to get the base tensor parallel plan for Llama3 model, which will also be shared with 3.1, 3.2, and 3.3 models

    Args:
        enable_float8 (bool): Whether float8 training is enabled and TP classes should use fp8 alternatives

    Returns:
        Dict[str, Any]: The tensor parallel plan for Llama3 model.
    """
    if enable_float8:
        rowwise_parallel, colwise_parallel = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
        )
    else:
        rowwise_parallel, colwise_parallel = (
            RowwiseParallel,
            ColwiseParallel,
        )

    # Define the Tensor Parallel plan for Llama3 model, which will also be shared with 3.1, 3.2, and 3.3 models
    base_llama_tp_plan = {
        "tok_embeddings": rowwise_parallel(input_layouts=Replicate()),
        "output": colwise_parallel(output_layouts=Replicate()),
        "layers.*.attn.q_proj": colwise_parallel(),
        "layers.*.attn.k_proj": colwise_parallel(),
        "layers.*.attn.v_proj": colwise_parallel(),
        "layers.*.attn.output_proj": rowwise_parallel(),
        "layers.*.mlp.w1": colwise_parallel(),
        "layers.*.mlp.w2": rowwise_parallel(),
        "layers.*.mlp.w3": colwise_parallel(),
    }
    return base_llama_tp_plan
