# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
    colwise_parallel_cls: type[ParallelStyle] = ColwiseParallel,
    rowwise_parallel_cls: type[ParallelStyle] = RowwiseParallel,
    prepare_module_input_cls: type[ParallelStyle] = PrepareModuleInput,
) -> dict[str, ParallelStyle]:
    """
    Define the Tensor Parallel plan for Llama3 model, which will also be shared with 3.1, 3.2, and 3.3 models.
    """
    return {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(1)
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate(),
        ),
        "layers.*.attn": prepare_module_input_cls(
            input_layouts=(Shard(1), Shard(1)),
            desired_input_layouts=(Replicate(), Replicate()),
        ),
        "layers.*.mlp": prepare_module_input_cls(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "layers.*.sa_norm": SequenceParallel(),
        "layers.*.mlp_norm": SequenceParallel(),
        "layers.*.attn.q_proj": colwise_parallel_cls(),
        "layers.*.attn.k_proj": colwise_parallel_cls(),
        "layers.*.attn.v_proj": colwise_parallel_cls(),
        "layers.*.attn.output_proj": rowwise_parallel_cls(output_layouts=Shard(1)),
        "layers.*.mlp.w1": colwise_parallel_cls(),
        "layers.*.mlp.w2": rowwise_parallel_cls(output_layouts=Shard(1)),
        "layers.*.mlp.w3": colwise_parallel_cls(),
    }


def _get_base_llama_tp_inference_plan():
    return {
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


def _get_fp8_llama_tp_training_plan(model: nn.Module) -> dict[str, ParallelStyle]:
    """
    Return the tensor parallel plan for Llama3 model that uses float8 for all-gather for both
    rowwise and colwise computation, currently only compatible with float8 fine-tuning with
    "tensorwise" scaling. This tensor parallel plan is shared between 3.1, 3.2, and 3.3 models.

    Args:
        model (nn.Module): Model to generate plan for (no-op)

    Returns:
        dict[str, Any]: The float8-enabled tensor parallel plan for Llama3 model.
    """
    return _get_base_llama_tp_training_plan(
        colwise_parallel_cls=Float8ColwiseParallel,
        rowwise_parallel_cls=Float8RowwiseParallel,
        prepare_module_input_cls=PrepareFloat8ModuleInput,
    )


def base_llama_tp_plan(
    model: nn.Module,
    *,
    inference: bool = False,
    enable_fp8_training: bool = False,
) -> dict[str, ParallelStyle]:
    """
    Helper function to get the base tensor parallel plan for Llama3 model, which will also be shared with 3.1, 3.2, and 3.3 models

    Args:
        model (nn.Module): Model to generate plan for (no-op)
        inference (bool): Whether running inference or not
        enable_fp8_training (bool): Whether to enable float8 training.

    Returns:
        dict[str, Any]: The tensor parallel plan for Llama3 model.

    Raises:
        ValueError: if FP8 training is enabled with inference.
    """
    if enable_fp8_training:
        if inference:
            raise ValueError(
                "FP8 training is not compatible with inference with LLaMA-3"
            )
        return _get_fp8_llama_tp_training_plan(model)

    return (
        _get_base_llama_tp_inference_plan()
        if inference
        else _get_base_llama_tp_training_plan()
    )
