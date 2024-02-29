# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Union

import torch
from torch.nn import Module

from torchtune.models import llama2

from torchtune.utils import get_device

ALL_MODELS = {"llama2_7b": llama2.llama2_7b, "lora_llama2_7b": llama2.lora_llama2_7b}
ALL_TOKENIZERS = {"llama2_tokenizer": llama2.llama2_tokenizer}


def get_model(name: str, device: Union[str, torch.device], **kwargs) -> Module:
    """Get known supported models by name"""
    if name in ALL_MODELS:
        with get_device(device):
            model = ALL_MODELS[name](**kwargs)
        return model
    else:
        raise ValueError(
            f"Model not recognized. Expected one of {ALL_MODELS}, received {name}"
        )


def get_tokenizer(name: str, **kwargs) -> Callable:
    """Get known supported tokenizers by name"""
    if name in ALL_TOKENIZERS:
        return ALL_TOKENIZERS[name](**kwargs)
    else:
        raise ValueError(
            f"Tokenizer not recognized. Expected one of {ALL_TOKENIZERS}, received {name}"
        )


def list_models():
    """List of availabe models supported by `get_model`"""
    return list(ALL_MODELS)


def list_tokenizers():
    """List of availabe tokenizers supported by `get_tokenizer`"""
    return list(ALL_TOKENIZERS)
