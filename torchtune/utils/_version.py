# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch


def torch_version_ge(version: str) -> bool:
    """
    Check if torch version is greater than or equal to the given version
    """
    return version in torch.__version__ or torch.__version__ >= version
