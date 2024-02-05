# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .llama2 import llama2_tokenizer

__all__ = [
    "llama2_tokenizer",
]


def list_tokenizers():
    """List of available tokenizers supported by `get_tokenizer`"""
    return __all__
