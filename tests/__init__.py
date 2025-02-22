# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Check at the top-level that torch, torchao, and torchvision are installed.
# This is better than doing it at every import site.
# We have to do this because it is not currently possible to
# properly support both nightly and stable installs of PyTorch + torchao
# in pyproject.toml.
try:
    import torch  # noqa
except ImportError as e:
    raise ImportError(
        """
        torch not installed.
        Please follow the instructions at https://pytorch.org/torchtune/main/install.html#pre-requisites
        to install torch.
        """
    ) from e
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
try:
    import torchvision  # noqa
except ImportError as e:
    raise ImportError(
        """
        torchvision not installed.
        Please follow the instructions at https://pytorch.org/torchtune/main/install.html#pre-requisites
        to install torchvision.
        """
    ) from e
