# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ._component_builders import llama3_1, lora_llama3_1

from ._model_builders import (  # noqa
    llama3_1_405b,
    llama3_1_70b,
    llama3_1_8b,
    lora_llama3_1_405b,
    lora_llama3_1_70b,
    lora_llama3_1_8b,
    qlora_llama3_1_405b,
    qlora_llama3_1_70b,
    qlora_llama3_1_8b,
)
from ._position_embeddings import Llama3ScaledRoPE

__all__ = [
    "llama3_1",
    "llama3_1_8b",
    "llama3_1_70b",
    "llama3_1_405b",
    "lora_llama3_1",
    "lora_llama3_1_8b",
    "lora_llama3_1_70b",
    "lora_llama3_1_405b",
    "qlora_llama3_1_8b",
    "qlora_llama3_1_70b",
    "qlora_llama3_1_405b",
    "Llama3ScaledRoPE",
]
