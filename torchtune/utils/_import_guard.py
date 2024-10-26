# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchao
from torchtune.utils._version import _is_fbcode, _nightly_version_ge, torch_version_ge

# We can only use flex attention / BlockMask if torch version >= 2.5.0 and GPU is Turing / SM75 and above
_SUPPORTS_FLEX_ATTENTION = (
    torch_version_ge("2.5.0")
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (7, 5)
)

torchao_version = torchao.__version__

_USE_NEW_TENSOR_CORE_TILED_LAYOUT_API = _is_fbcode() or (
    not _is_fbcode()
    and (
        ("dev" not in torchao_version and torchao_version >= "0.7.0")
        or (
            "dev" in torchao_version
            and _nightly_version_ge(torchao_version, "2024-10-10")
        )
    )
)
