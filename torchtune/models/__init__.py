# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .llama2 import llama2_7b
from .lora_llama2 import lora_llama2, lora_llama2_7b

__all__ = ["llama2_7b", "lora_llama2", "lora_llama2_7b"]


def list_models():
    """List of available models supported by `get_model`"""
    return __all__
