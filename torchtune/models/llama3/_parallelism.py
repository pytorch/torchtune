# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Type

from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.parallel.style import ParallelStyle

from torchao.float8.float8_tensor_parallel import (
    Float8ColwiseParallel,
    Float8RowwiseParallel,
)


def _get_base_llama_tp_plan(
    _sequence_parallel_cls: Type[ParallelStyle] = SequenceParallel,
    _colwise_parallel_cls: Type[ParallelStyle] = ColwiseParallel,
    _rowwise_parallel_cls: Type[ParallelStyle] = RowwiseParallel,
) -> Dict[str, ParallelStyle]:
    """
    Define the Tensor Parallel plan for Llama3 model, which will also be shared with 3.1, 3.2, and 3.3 models.
    """
    return {
        "tok_embeddings": _rowwise_parallel_cls(
            input_layouts=Replicate(), output_layouts=Shard(1)
        ),
        "norm": _sequence_parallel_cls(),
        "output": _colwise_parallel_cls(
            input_layouts=Shard(1), output_layouts=Replicate()
        ),
        "layers.*.attn": PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        ),
        "layers.*.mlp": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "layers.*.sa_norm": _sequence_parallel_cls(),
        "layers.*.mlp_norm": _sequence_parallel_cls(),
        "layers.*.attn.q_proj": _colwise_parallel_cls(),
        "layers.*.attn.k_proj": _colwise_parallel_cls(),
        "layers.*.attn.v_proj": _colwise_parallel_cls(),
        "layers.*.attn.output_proj": _rowwise_parallel_cls(output_layouts=Shard(1)),
        "layers.*.mlp.w1": _colwise_parallel_cls(),
        "layers.*.mlp.w2": _rowwise_parallel_cls(output_layouts=Shard(1)),
        "layers.*.mlp.w3": _colwise_parallel_cls(),
    }


_BASE_LLAMA_TP_PLAN = _get_base_llama_tp_plan()
_FP8_LLAMA_TP_PLAN = _get_base_llama_tp_plan(
    _colwise_parallel_cls=Float8ColwiseParallel,
    _rowwise_parallel_cls=Float8RowwiseParallel,
)


def base_llama_tp_plan() -> Dict[str, ParallelStyle]:
    """
    Helper function to get the base tensor parallel plan for Llama3 model, which will also be shared with 3.1, 3.2, and 3.3 models

    Returns:
        Dict[str, Any]: The tensor parallel plan for Llama3 model.
    """
    return _BASE_LLAMA_TP_PLAN


def fp8_llama_tp_plan() -> Dict[str, ParallelStyle]:
    """
    Return the tensor parallel plan for Llama3 model that uses float8 for all-gather for both
    rowwise and colwise computation, currently only compatible with float8 fine-tuning with
    "tensorwise" scaling. This tensor parallel plan is shared between 3.1, 3.2, and 3.3 models.

    Returns:
        Dict[str, Any]: The float8-enabled tensor parallel plan for Llama3 model.
    """
    return _FP8_LLAMA_TP_PLAN
