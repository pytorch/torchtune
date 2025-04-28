# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Optional

from torch import nn
from torch.distributed.tensor.parallel.style import ParallelStyle

from torchao.dtypes import TensorCoreTiledLayout
from torchao.float8 import (
    convert_to_float8_training as _convert_to_float8_training_torchao,
    Float8LinearConfig,
)
from torchao.float8.float8_tensor_parallel import (
    Float8ColwiseParallel,
    Float8RowwiseParallel,
)
from torchao.quantization import (
    int4_weight_only,
    int8_dynamic_activation_int4_weight,
    quantize_,
)
from torchao.quantization.qat import (
    Int4WeightOnlyQATQuantizer,
    Int8DynActInt4WeightQATQuantizer,
)
from torchao.quantization.qat.linear import (
    disable_4w_fake_quant,
    disable_8da4w_fake_quant,
    enable_4w_fake_quant,
    enable_8da4w_fake_quant,
)

from torchtune.modules.peft.lora import LoRALinear, QATLoRALinear


__all__ = [
    "get_quantizer_mode",
    "Int4WeightOnlyQuantizer",
    "Int4WeightOnlyQATQuantizer",
    "Int4WeightOnlyQATQuantizerModuleSwap",
    "Int8DynActInt4WeightQuantizer",
    "Int8DynActInt4WeightQATQuantizer",
    "Int8DynActInt4WeightQATQuantizerModuleSwap",
]


try:
    from torchao.quantization import qat  # noqa: F401
except ImportError as e:
    raise ValueError("Need torchao version 0.7.0+") from e

_quantizer_to_mode = {}
_quantizer_mode_to_disable_fake_quant = {}
_quantizer_mode_to_enable_fake_quant = {}


# ========================================
# int8 dynamic activations + int4 weight |
# ========================================


class Int8DynActInt4WeightQuantizer:
    """
    Quantizer for applying int8 per token dynamic activation + int4
    per group weight quantization to linear layers in the model.
    """

    def __init__(self, groupsize: int = 256):
        self.groupsize = groupsize

    def quantize(self, model):
        quantize_fn = int8_dynamic_activation_int4_weight(self.groupsize)
        quantize_(model, quantize_fn)
        return model


_quantizer_to_mode[Int8DynActInt4WeightQuantizer] = "8da4w"
_quantizer_to_mode[Int8DynActInt4WeightQATQuantizer] = "8da4w-qat"
_quantizer_mode_to_disable_fake_quant["8da4w-qat"] = disable_8da4w_fake_quant
_quantizer_mode_to_enable_fake_quant["8da4w-qat"] = enable_8da4w_fake_quant


# ==================
# int4 weight only |
# ==================


class Int4WeightOnlyQuantizer:
    """
    Quantizer for applying int4 per group weight only quantization
    to linear layers in the model using the efficient tinygemm kernel.
    """

    def __init__(self, groupsize: int = 128, inner_k_tiles: int = 8):
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles

    def quantize(self, model):
        layout_type = TensorCoreTiledLayout(self.inner_k_tiles)
        quantize_fn = int4_weight_only(self.groupsize, layout_type)
        quantize_(model, quantize_fn)
        return model


_quantizer_to_mode[Int4WeightOnlyQuantizer] = "4w"
_quantizer_to_mode[Int4WeightOnlyQATQuantizer] = "4w-qat"
_quantizer_mode_to_disable_fake_quant["4w-qat"] = disable_4w_fake_quant
_quantizer_mode_to_enable_fake_quant["4w-qat"] = enable_4w_fake_quant


# ====================== #
# Backward compatibility #
# ====================== #


# int4 weight-only
class Int4WeightOnlyQATQuantizerModuleSwap(Int4WeightOnlyQATQuantizer):
    pass


disable_4w_fake_quant_module_swap = disable_4w_fake_quant
enable_4w_fake_quant_module_swap = enable_4w_fake_quant
_quantizer_to_mode[Int4WeightOnlyQATQuantizerModuleSwap] = "4w-qat-module-swap"
_quantizer_mode_to_disable_fake_quant[
    "4w-qat-module-swap"
] = disable_4w_fake_quant_module_swap
_quantizer_mode_to_enable_fake_quant[
    "4w-qat-module-swap"
] = enable_4w_fake_quant_module_swap


# int8 dynamic activations + int4 weight
class Int8DynActInt4WeightQATQuantizerModuleSwap(Int8DynActInt4WeightQATQuantizer):
    pass


disable_8da4w_fake_quant_module_swap = disable_8da4w_fake_quant
enable_8da4w_fake_quant_module_swap = enable_8da4w_fake_quant
_quantizer_to_mode[Int8DynActInt4WeightQATQuantizerModuleSwap] = "8da4w-qat-module-swap"
_quantizer_mode_to_disable_fake_quant[
    "8da4w-qat-module-swap"
] = disable_8da4w_fake_quant_module_swap
_quantizer_mode_to_enable_fake_quant[
    "8da4w-qat-module-swap"
] = enable_8da4w_fake_quant_module_swap


def get_quantizer_mode(quantizer: Optional[Callable]) -> Optional[str]:
    """Given a quantizer object, returns a string that specifies the type of quantization.

    For example, in the case of int4 weight only quantization, we'll return "4w".
    If the quantizer is not recognized as a known quantizer, we'll return None.

    Currently supported:

    - :class:`~torchtune.training.quantization.Int8DynActInt4WeightQuantizer`: "8da4w"
    - :class:`~torchtune.training.quantization.Int4WeightOnlyQuantizer`: "4w"
    - :class:`~torchao.quantization.qat.Int8DynActInt4WeightQATQuantizer`: "8da4w-qat"
    - :class:`~torchao.quantization.qat.Int4WeightOnlyQATQuantizer`: "4w-qat"

    Args:
        quantizer (Optional[Callable]): A callable object that implements the `quantize` method.

    Returns:
        Optional[str]: The quantization mode.
    """
    return _quantizer_to_mode.get(type(quantizer), None)


def _get_disable_fake_quant(quantizer_mode: str) -> Callable:
    """Given a quantizer mode, return the corresponding function for disabling fake
    quantize in a model prepared by the quantizer.
    If the quantizer is not recognized as a known QAT quantizer, return None.
    """
    return _quantizer_mode_to_disable_fake_quant.get(quantizer_mode, None)


def _get_enable_fake_quant(quantizer_mode: str) -> Callable:
    """Given a quantizer mode, return the corresponding function for enabling fake
    quantize in a model prepared by the quantizer.
    If the quantizer is not recognized as a known QAT quantizer, return None.
    """
    return _quantizer_mode_to_enable_fake_quant.get(quantizer_mode, None)


def swap_lora_linear_with_qat(
    module: nn.Module,
    # TODO: make the types Optional[FakeQuantizeConfig] once we
    # support torchao 0.7+ by default
    activation_qat_config: Optional["FakeQuantizeConfig"] = None,
    weight_qat_config: Optional["FakeQuantizeConfig"] = None,
) -> None:
    """
    Swap all `LoRALinear` in the model with `QATLoRALinear`.

    This is used for combining QAT + LoRA during finetuning. The resulting linear layers
    will apply the following transformation instead:

        x -> fake_quantize(W_frozen) @ fake_quantize(x) + BAx

    Fake quantization here refers to simulating the quantization numerics without actual
    dtype casting, with the goal of providing improved accuracies when the model is
    ultimately quantized after finetuning.

    Args:
        module (nn.Module): The model to swap linear layers on
        activation_qat_config (Optional[FakeQuantizeConfig]): The config for specifying
            how to fake quantize input activations in the base linear layer
        weight_qat_config (Optional[FakeQuantizeConfig]): The config for specifying
            how to fake quantize base linear weights
    """
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            new_linear = QATLoRALinear.from_lora_linear(
                child,
                activation_qat_config,
                weight_qat_config,
            )
            setattr(module, name, new_linear)
        else:
            swap_lora_linear_with_qat(
                child,
                activation_qat_config,
                weight_qat_config,
            )


def convert_to_float8_training(
    model: nn.Module,
    fp8_recipe_name: Optional[str] = None,
) -> nn.Module:
    """
    Prepare the model for float8 training by swapping all `nn.Linear` with `Float8Linear`.

    Args:
        model (nn.Module): The model to swap linear layers on
        fp8_recipe_name (Optional[str]): name to identify one of the pre-made recipes,
            one of "tensorwise", "rowwise", and "rowwise_with_gw_hp". If not specified,
            defaults to "tensorwise" with "enable_fsdp_float8_all_gather=True". See
            https://github.com/pytorch/ao/blob/v0.9.0/torchao/float8/config.py#L150
            for more details.

    Returns:
        (nn.Module) The new model with `Float8Linear`.
    """
    if fp8_recipe_name is not None:
        fp8_config = Float8LinearConfig.from_recipe_name(fp8_recipe_name)
    else:
        fp8_config = Float8LinearConfig(enable_fsdp_float8_all_gather=True)
    return _convert_to_float8_training_torchao(
        model,
        config=fp8_config,
        module_filter_fn=lambda mod, fqn: fqn != "output",
    )


# TODO: validate this in full_finetune_distributed recipe once FP8 + TP is enabled
def _validate_float8_tp_plan(
    tp_plan: Optional[Dict[str, ParallelStyle]],
    fp8_recipe_name: Optional[str] = None,
) -> None:
    """
    Validate that the provided tensor parallel plan is compatible with the
    float8 settings. Specifically, float8 tensor parallel plans are only
    supported when using 'tensorwise' float8 recipes.
    """
    if tp_plan is None or is_fp8_tensorwise_scaling(fp8_recipe_name):
        return
    for parallel_style in tp_plan.values():
        if isinstance(parallel_style, Float8ColwiseParallel) or isinstance(
            parallel_style, Float8RowwiseParallel
        ):
            raise ValueError(
                "%s and %s are only compatible with 'tensorwise' float8 recipes"
                % (Float8ColwiseParallel.__name__, Float8RowwiseParallel.__name__)
            )


def is_fp8_tensorwise_scaling(fp8_recipe_name: Optional[str]):
    """
    Return True if the fp8 recipe name refers to 'tensorwwise' scaling.
    """
    return fp8_recipe_name is None or fp8_recipe_name == "tensorwise"
