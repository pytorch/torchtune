# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__version__ = ""


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
    If you are using stable PyTorch please install via pip install torchao.
    If you are using nightly PyTorch please install via
    pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu121."
    """
    ) from e

from torchtune import datasets, models, modules, utils

__all__ = [datasets, models, modules, utils]
