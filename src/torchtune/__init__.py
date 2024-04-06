# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune import datasets, models, modules, utils

__all__ = [datasets, models, modules, utils]

# Import version from version.txt
with open("../../version.txt", "r") as f:
    __version__ = f.read().strip()
