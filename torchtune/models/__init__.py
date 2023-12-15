# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
from torch.nn import Module

from torchtune.models.llama2.models import llama2_7b, llama2_tokenizer

MODEL_DICT = {"llama2_7b": llama2_7b}
TOKENIZER_DICT = {"llama2_tokenizer": llama2_tokenizer}


def get_model(name: str, device: str, **kwargs) -> Module:
    if name in MODEL_DICT:
        with torch.device(device):
            model = MODEL_DICT[name](**kwargs)
        return model
    else:
        raise ValueError(f"Unknown model: {name}")


def get_tokenizer(name: str, **kwargs) -> Callable:
    if name in TOKENIZER_DICT:
        return TOKENIZER_DICT[name](**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer: {name}")


def list_models():
    return list(MODEL_DICT)


def list_tokenizers():
    return list(TOKENIZER_DICT)
