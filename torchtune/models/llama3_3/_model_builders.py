# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torchtune.models.llama3_1._model_builders import (
    llama3_1_70b,
    lora_llama3_1_70b,
    qlora_llama3_1_70b,
)

"""
Model builders build specific instantiations using component builders. The Llama3.3 model
builders all call the Llama3.1 models as they're identical models apart from the checkpoints.
"""

llama3_3_70b = llama3_1_70b

llama3_3_70b.__doc__ = """
Builder for creating a Llama3.3 model initialized w/ the default 70B parameter values.
Please see `llama3_1_70b` for full API arguments.
"""

lora_llama3_3_70b = lora_llama3_1_70b

lora_llama3_3_70b.__doc__ = """
Builder for creating a Llama3.3 70B model with LoRA enabled.
Please see `lora_llama3_1_70b` for full API arguments.
"""

qlora_llama3_3_70b = qlora_llama3_1_70b

qlora_llama3_1_70b.__doc__ = """
Builder for creating a Llama3.3 70B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_llama3_1_70b` for full API arguments.
"""
