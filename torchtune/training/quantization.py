# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

__all__ = [
    "get_quantizer_mode",
]


_quantizer_to_mode = {}
_quantizer_mode_to_disable_fake_quant = {}
_quantizer_mode_to_enable_fake_quant = {}


from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

__all__.append("Int8DynActInt4WeightQuantizer")
_quantizer_to_mode[Int8DynActInt4WeightQuantizer] = "8da4w"


from torchao.quantization.prototype.qat import (
    disable_8da4w_fake_quant,
    enable_8da4w_fake_quant,
    Int8DynActInt4WeightQATQuantizer,
)

__all__.append("Int8DynActInt4WeightQATQuantizer")
_quantizer_to_mode[Int8DynActInt4WeightQATQuantizer] = "8da4w-qat"
_quantizer_mode_to_disable_fake_quant["8da4w-qat"] = disable_8da4w_fake_quant
_quantizer_mode_to_enable_fake_quant["8da4w-qat"] = enable_8da4w_fake_quant

try:
    # Note: QAT tensor subclass implementation in torchao only works
    # with FSDP2 today. For other distribution strategies like DDP and
    # FSDP1, users will need to fall back to the old module swap flow.
    # TODO: remove this try catch once we upgrade to torchao 0.5.0

    from torchao.quantization.prototype.qat._module_swap_api import (
        disable_8da4w_fake_quant_module_swap,
        enable_8da4w_fake_quant_module_swap,
        Int8DynActInt4WeightQATQuantizerModuleSwap,
    )

    __all__.append("Int8DynActInt4WeightQATQuantizerModuleSwap")
    _quantizer_to_mode[
        Int8DynActInt4WeightQATQuantizerModuleSwap
    ] = "8da4w-qat-module-swap"
    _quantizer_mode_to_disable_fake_quant[
        "8da4w-qat-module-swap"
    ] = disable_8da4w_fake_quant_module_swap
    _quantizer_mode_to_enable_fake_quant[
        "8da4w-qat-module-swap"
    ] = enable_8da4w_fake_quant_module_swap
except ImportError:
    pass


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
