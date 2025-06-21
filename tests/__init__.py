# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import warnings

# Avoid memory fragmentation and peak reserved memory increasing over time
# To overwrite, set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    if "torch" in sys.modules:
        warnings.warn(
            "The 'torch' module has already been imported. "
            "Setting PYTORCH_CUDA_ALLOC_CONF may not have an effect."
            "For best results, set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before importing 'torch'."
        )
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Check at the top-level that torchao is installed.
# This is better than doing it at every import site.
# We have to do this because it is not currently possible to
# properly support both nightly and stable installs of PyTorch + torchao
# in pyproject.toml.
try:
    import torchao  # noqa
except ImportError as e:
    raise ImportError(
        """
        torchao not installed.
        Please follow the instructions at https://pytorch.org/torchtune/main/install.html#pre-requisites
        to install torchao.
        """
    ) from e
