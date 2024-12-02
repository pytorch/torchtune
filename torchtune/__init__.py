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
        Please follow the instructions at https://pytorch.org/torchtune/main/install.html#pre-requisites
        to install torchao.
        """
    ) from e

# Enables faster downloading. For more info: https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
# To disable, run `HF_HUB_ENABLE_HF_TRANSFER=0 tune download <model_config>`
try:
    import os

    import hf_transfer  # noqa

    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass

from torchtune import datasets, generation, models, modules, utils

__all__ = [datasets, models, modules, utils, generation]
