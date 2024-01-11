# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Union

import torch
from torch.nn import Module

from torchtune.models.llama2.models import llama2_7b, llama2_tokenizer
from torchtune.utils import get_device

_MODEL_DICT = {"llama2_7b": llama2_7b}
_TOKENIZER_DICT = {"llama2_tokenizer": llama2_tokenizer}


def get_model(name: str, device: Union[str, torch.device], **kwargs) -> Module:
    """Get known supported models by name"""
    if name in _MODEL_DICT:
        with get_device(device):
            model = _MODEL_DICT[name](**kwargs)
        return model
    else:
        raise ValueError(f"Unknown model: {name}")


def get_tokenizer(name: str, **kwargs) -> Callable:
    """Get known supported tokenizers by name"""
    if name in _TOKENIZER_DICT:
        return _TOKENIZER_DICT[name](**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer: {name}")


def list_models():
    """List of availabe models supported by `get_model`"""
    return list(_MODEL_DICT)


def list_tokenizers():
    """List of availabe tokenizers supported by `get_tokenizer`"""
    return list(_TOKENIZER_DICT)
