# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Type

from torch import nn

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
    PrepareFloat8ModuleInput,
)


def _get_base_llama_tp_training_plan(
    layerwise_colwise_parallel_cls: Type[ParallelStyle] = ColwiseParallel,
    layerwise_rowwise_parallel_cls: Type[ParallelStyle] = RowwiseParallel,
    layerwise_prepare_module_input_cls: Type[ParallelStyle] = PrepareModuleInput,
) -> Dict[str, ParallelStyle]:
    """
    Define the Tensor Parallel plan for Llama3 model, which will also be shared with 3.1, 3.2, and 3.3 models.
    """
    return {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(1)
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
        "layers.*.attn": layerwise_prepare_module_input_cls(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        ),
        "layers.*.mlp": layerwise_prepare_module_input_cls(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "layers.*.sa_norm": SequenceParallel(),
        "layers.*.mlp_norm": SequenceParallel(),
        "layers.*.attn.q_proj": layerwise_colwise_parallel_cls(),
        "layers.*.attn.k_proj": layerwise_colwise_parallel_cls(),
        "layers.*.attn.v_proj": layerwise_colwise_parallel_cls(),
        "layers.*.attn.output_proj": layerwise_rowwise_parallel_cls(
            output_layouts=Shard(1)
        ),
        "layers.*.mlp.w1": layerwise_colwise_parallel_cls(),
        "layers.*.mlp.w2": layerwise_rowwise_parallel_cls(output_layouts=Shard(1)),
        "layers.*.mlp.w3": layerwise_colwise_parallel_cls(),
    }


BASE_LLAMA_TP_TRAINING_PLAN = _get_base_llama_tp_training_plan()

FP8_LLAMA_TP_TRAINING_PLAN = _get_base_llama_tp_training_plan(
    layerwise_colwise_parallel_cls=Float8ColwiseParallel,
    layerwise_rowwise_parallel_cls=Float8RowwiseParallel,
    layerwise_prepare_module_input_cls=PrepareFloat8ModuleInput,
)

BASE_LLAMA_TP_INFERENCE_PLAN = {
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


def base_llama_tp_plan(
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
    return BASE_LLAMA_TP_INFERENCE_PLAN if inference else BASE_LLAMA_TP_TRAINING_PLAN


# TODO: expose this once tested
def _fp8_llama_tp_plan() -> Dict[str, ParallelStyle]:
    """
    Return the tensor parallel plan for Llama3 model that uses float8 for all-gather for both
    rowwise and colwise computation, currently only compatible with float8 fine-tuning with
    "tensorwise" scaling. This tensor parallel plan is shared between 3.1, 3.2, and 3.3 models.

    Returns:
        Dict[str, Any]: The float8-enabled tensor parallel plan for Llama3 model.
    """
    return FP8_LLAMA_TP_TRAINING_PLAN
