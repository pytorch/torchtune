# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

# We can only use flex attention / BlockMask if torch version >= 2.5.0 and GPU is Turing / SM75 and above
_SUPPORTS_FLEX_ATTENTION = (
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 5)
)
