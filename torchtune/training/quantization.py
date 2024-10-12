# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from torchtune.utils._import_guard import _USE_NEW_TENSOR_CORE_TILED_LAYOUT_API

if _USE_NEW_TENSOR_CORE_TILED_LAYOUT_API:
    from torchao.dtypes import TensorCoreTiledLayout
else:
    from torchao.dtypes import TensorCoreTiledLayoutType as TensorCoreTiledLayout

from torchao.quantization import (
    int4_weight_only,
    int8_dynamic_activation_int4_weight,
    quantize_,
)
from torchao.quantization.prototype.qat import (
    disable_4w_fake_quant,
    disable_8da4w_fake_quant,
    enable_4w_fake_quant,
    enable_8da4w_fake_quant,
    Int4WeightOnlyQATQuantizer,
    Int8DynActInt4WeightQATQuantizer,
)
from torchao.quantization.prototype.qat._module_swap_api import (
    disable_4w_fake_quant_module_swap,
    disable_8da4w_fake_quant_module_swap,
    enable_4w_fake_quant_module_swap,
    enable_8da4w_fake_quant_module_swap,
    Int4WeightOnlyQATQuantizerModuleSwap,
    Int8DynActInt4WeightQATQuantizerModuleSwap,
)


__all__ = [
    "get_quantizer_mode",
    "Int4WeightOnlyQuantizer",
    "Int4WeightOnlyQATQuantizer",
    "Int4WeightOnlyQATQuantizerModuleSwap",
    "Int8DynActInt4WeightQuantizer",
    "Int8DynActInt4WeightQATQuantizer",
    "Int8DynActInt4WeightQATQuantizerModuleSwap",
]


_quantizer_to_mode = {}
_quantizer_mode_to_disable_fake_quant = {}
_quantizer_mode_to_enable_fake_quant = {}


# ========================================================
# int8 dynamic activations + int4 weight tensor subclass |
# ========================================================


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


# =============
# module swap |
# =============

# Note: QAT tensor subclass implementation in torchao only works
# with FSDP2 today. For other distribution strategies like DDP and
# FSDP1, users will need to fall back to the old module swap flow.

# int4 weight-only
_quantizer_to_mode[Int4WeightOnlyQATQuantizerModuleSwap] = "4w-qat-module-swap"
_quantizer_mode_to_disable_fake_quant[
    "4w-qat-module-swap"
] = disable_4w_fake_quant_module_swap
_quantizer_mode_to_enable_fake_quant[
    "4w-qat-module-swap"
] = enable_4w_fake_quant_module_swap

# int8 dynamic activations + int4 weight
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

    - :class:`~torchao.quantization.quant_api.Int8DynActInt4WeightQuantizer`: "8da4w" (requires ``torch>=2.3.0``)
    - :class:`~torchao.quantization.prototype.qat.Int8DynActInt4WeightQATQuantizer`: "8da4w-qat" (requires ``torch>=2.4.0``)

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
