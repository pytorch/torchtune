# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional

import torch
from torchao.quantization.quant_api import (
    Int4WeightOnlyGPTQQuantizer,
    Int4WeightOnlyQuantizer,
    Quantizer,
)

from torchtune.modules.low_precision._utils import (
    _get_torchao_version,
    _nightly_version_ge,
)

ao_version, is_nightly = _get_torchao_version()
if is_nightly and _nightly_version_ge(ao_version, "2024-07-03"):
    from torchao.quantization.quant_api import quantize_ as quantize
else:
    from torchao.quantization.quant_api import quantize

# importing TORCH_VERSION_AFTER_2_3 because `Int8DynActInt4WeightQuantizer`
# is only available after 2.3 so we have to guard the pytorch versions to decide
# the list of supported quantizers
from torchao.utils import TORCH_VERSION_AFTER_2_3, TORCH_VERSION_AFTER_2_4

__all__ = [
    "Int4WeightOnlyQuantizer",
    "Int4WeightOnlyGPTQQuantizer",
    "Int8WeightOnlyQuantizer",
    "get_quantizer_mode",
]


class Int8WeightOnlyQuantizer(Quantizer):
    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        return quantize(model, "int8_weight_only")


_quantizer_to_mode = {
    Int4WeightOnlyQuantizer: "4w",
    Int8WeightOnlyQuantizer: "8w",
    Int4WeightOnlyGPTQQuantizer: "4w-gptq",
}
_quantizer_mode_to_disable_fake_quant = {}
_quantizer_mode_to_enable_fake_quant = {}


if TORCH_VERSION_AFTER_2_3:
    from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

    __all__.append("Int8DynActInt4WeightQuantizer")
    _quantizer_to_mode[Int8DynActInt4WeightQuantizer] = "8da4w"


if TORCH_VERSION_AFTER_2_4:
    from torchao.quantization.prototype.qat import (
        disable_8da4w_fake_quant,
        enable_8da4w_fake_quant,
        Int8DynActInt4WeightQATQuantizer,
    )

    __all__.append("Int8DynActInt4WeightQATQuantizer")
    _quantizer_to_mode[Int8DynActInt4WeightQATQuantizer] = "8da4w-qat"
    _quantizer_mode_to_disable_fake_quant["8da4w-qat"] = disable_8da4w_fake_quant
    _quantizer_mode_to_enable_fake_quant["8da4w-qat"] = enable_8da4w_fake_quant


def get_quantizer_mode(quantizer: Optional[Callable]) -> Optional[str]:
    """Given a quantizer object, returns a string that specifies the type of quantization.

    For example, in the case of int4 weight only quantization, we'll return "4w".
    If the quantizer is not recognized as a known quantizer, we'll return None.

    Currently supported:

    - :class:`~torchao.quantization.quant_api.Int4WeightOnlyQuantizer`: "4w"
    - :class:`~torchao.quantization.quant_api.Int8WeightOnlyQuantizer`: "8w"
    - :class:`~torchao.quantization.quant_api.Int4WeightOnlyGPTQQuantizer`: "4w-gptq"
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
