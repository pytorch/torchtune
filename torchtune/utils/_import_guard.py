# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import torch

# We can only use flex attention / BlockMask if torch version >= 2.5.0 and GPU is Turing / SM75 and above
_SUPPORTS_FLEX_ATTENTION = (
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 5)
)

_TORCHDATA_MIN_VERSION = "0.10.0"
if (
    importlib.util.find_spec("torchdata") is not None
    and importlib.util.find_spec("torchdata.nodes") is not None
):
    _TORCHDATA_INSTALLED = True
else:
    _TORCHDATA_INSTALLED = False
