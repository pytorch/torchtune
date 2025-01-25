# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa: F401

# NOTE: This file is maintained for backward compatibility purposes.
# The imports below point to the new location in `torchtune.modules.transforms.tokenizers`.
# The import paths will be removed in v0.7. Please update your code to use the new path
# (torchtune.modules.transforms.tokenizers) to avoid breaking changes in future releases.


import warnings

from torchtune.modules.transforms.tokenizers import (
    BaseTokenizer,
    ModelTokenizer,
    parse_hf_tokenizer_json,
    SentencePieceBaseTokenizer,
    TikTokenBaseTokenizer,
    tokenize_messages_no_special_tokens,
)

warnings.warn(
    "The import path 'torchtune.modules.tokenizers' is deprecated and will be removed in v0.7. "
    "Please update your imports to 'torchtune.modules.transforms.tokenizers'.",
    DeprecationWarning,
    stacklevel=2,
)
