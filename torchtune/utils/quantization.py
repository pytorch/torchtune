from typing import Any
import torch
from torchao.quantization.quant_api import (
    change_linear_weights_to_int4_woqtensors,
    change_linear_weights_to_int8_woqtensors,
    Quantizer,
)
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_3

class Int4WeightOnlyQuantizer(Quantizer):
    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        change_linear_weights_to_int4_woqtensors(model)
        return model


class Int8WeightOnlyQuantizer(Quantizer):
    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        change_linear_weights_to_int8_woqtensors(model)
        return model


def get_quantizer(quantization_mode, *args, **kwargs):
    qmode_to_quantizer = {
        # TODO: change to 4w before land
        "4w": Int4WeightOnlyQuantizer,
        "8w": Int8WeightOnlyQuantizer,
    }
    if TORCH_VERSION_AFTER_2_3:
        from torchao.quantization.quant_api import (
            Int8DynActInt4WeightQuantizer,
            Int8DynActInt4WeightGPTQQuantizer,
            Int4WeightOnlyGPTQQuantizer,
        )

        qmode_to_quantizer |= {
            "8da4w": Int8DynActInt4WeightQuantizer,
            # TODO: merge into 8da4w
            "8da4w-gptq": Int8DynActInt4WeightGPTQQuantizer,
            # merge into 4w
            "4w-gptq": Int4WeightOnlyGPTQQuantizer,
        }
    if quantization_mode not in qmode_to_quantizer:
        raise ValueError(f"Unsupported quantization mode: {quantization_mode}, supported modes are: {qmode_to_quantizer.keys()}")
    return qmode_to_quantizer[quantization_mode](*args, **kwargs)
