# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
from torch import nn
from torchao.quantization import int8_dynamic_activation_int4_weight, quantize_
from torchao.quantization.prototype.qat import (
    disable_8da4w_fake_quant,
    enable_8da4w_fake_quant,
    Int8DynActInt4WeightQATQuantizer,
)
from torchao.quantization.prototype.qat._module_swap_api import (
    disable_8da4w_fake_quant_module_swap,
    enable_8da4w_fake_quant_module_swap,
    Int8DynActInt4WeightQATQuantizerModuleSwap,
)

from torchtune.modules import TransformerDecoder
from torchtune.modules.low_precision._utils import _get_torchao_version
from torchtune.modules.peft import DoRALinear, LoRALinear
from torchtune.utils._version import torch_version_ge


_TORCHAO_VERSION, _ = _get_torchao_version()

_SUPPORTS_INT8_MIXED_PRECISION_TRAINING = (
    torch_version_ge("2.4.0")
    and tuple(_TORCHAO_VERSION.split(".")) >= ("0", "5", "0")
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (8, 0)
)

if _SUPPORTS_INT8_MIXED_PRECISION_TRAINING:
    from torchao.prototype.quantized_training import (
        int8_mixed_precision_training,
        Int8MixedPrecisionTrainingConfig,
    )


__all__ = [
    "get_quantizer_mode",
    "Int8DynActInt4WeightQuantizer",
    "Int8DynActInt4WeightQATQuantizer",
    "Int8MixedPrecisionTrainingQuantizer",
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


# ====================================================
# int8 dynamic activations + int4 weight module swap |
# ====================================================

# Note: QAT tensor subclass implementation in torchao only works
# with FSDP2 today. For other distribution strategies like DDP and
# FSDP1, users will need to fall back to the old module swap flow.
__all__.append("Int8DynActInt4WeightQATQuantizerModuleSwap")
_quantizer_to_mode[Int8DynActInt4WeightQATQuantizerModuleSwap] = "8da4w-qat-module-swap"
_quantizer_mode_to_disable_fake_quant[
    "8da4w-qat-module-swap"
] = disable_8da4w_fake_quant_module_swap
_quantizer_mode_to_enable_fake_quant[
    "8da4w-qat-module-swap"
] = enable_8da4w_fake_quant_module_swap


class Int8MixedPrecisionTrainingQuantizer:
    """Apply INT8 mixed-precision training. This only affects weights of ``nn.Linear``
    modules. During training, weights and activations are dynamically quantized to INT8
    to utilize fast matrix multiplication with INT8 tensor cores. This is also done in
    the backward pass.

    The expected end2end speedup is 40% on a single A100 and 70% on a single 4090, with
    minimal accuracy loss. If convergence is an issue, please refer to torchao
    documentation below.

    For more details, as well as details about arguments of this quantizer, please refer to
    https://github.com/pytorch/ao/tree/main/torchao/prototype/quantized_training#int8-mixed-precision

    Args:
        output (bool): whether to apply INT8 mixed-precision for calculating output.
        grad_input (bool): whether to apply INT8 mixed-precision for calculating grad_input.
        grad_weight (bool): whether to apply INT8 mixed-precision for calculating grad_weight.

    Raises:
        RuntimeError: If runtime requirements for INT8 mixed-precision training are not met.

    NOTE: Due to the limitations of the current implementation, the following
    requirements must be satisfied to enjoy the expected speedup:

    1. Must use ``torch.compile()`` (set ``compile=True``).
    2. Inputs to the model must not be too dynamic. For example, when input tokens
    length changes for every batch, you won't see the expected speedup.

    To satisfy (2), you can use :class:`~torchtune.datasets.PackedDataset` (set
    ``dataset.packed=True`` and ``tokenizer.max_seq_len`` to a desired value.), which
    ensures input tokens always have fixed length.
    """

    def __init__(
        self,
        output: bool = True,
        grad_input: bool = True,
        grad_weight: bool = True,
    ) -> None:
        if not _SUPPORTS_INT8_MIXED_PRECISION_TRAINING:
            raise RuntimeError(
                "INT8 mixed-precision training requires torch>=2.4, torchao>=0.5, and"
                " a CUDA capable device with compute capability >= 8.0"
            )

        self._config = Int8MixedPrecisionTrainingConfig(
            output=output,
            grad_input=grad_input,
            grad_weight=grad_weight,
        )

    def prepare(self, model: nn.Module) -> nn.Module:
        quantize_fn = int8_mixed_precision_training(self._config)

        # custom filter_fn to work with torchtune's peft
        def filter_fn(module: nn.Module, name: str) -> bool:
            if isinstance(module, nn.Linear):
                return not (name.endswith(".lora_a") or name.endswith(".lora_b"))

            return isinstance(module, (LoRALinear, DoRALinear))

        # Don't apply INT8 mixed-precision training to LM head since end2end speedup
        # will be slightly worse. There are also possible issues with tied word
        # embeddings.
        if isinstance(model, TransformerDecoder):
            quantize_(model.layers, quantize_fn, filter_fn=filter_fn)
        else:
            quantize_(model, quantize_fn, filter_fn=filter_fn)
        return model


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
