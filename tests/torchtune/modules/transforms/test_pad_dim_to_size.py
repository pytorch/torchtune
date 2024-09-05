# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from torchtune.modules.transforms.vision_utils.pad_dim_to_size import pad_dim_to_size


def test_pad_dim_to_size():
    image = torch.ones(2, 2, 2, 2, dtype=torch.float16)
    image = pad_dim_to_size(image, 4, 1)
    assert image.shape == (2, 4, 2, 2)
    assert image.mean() == 0.5, "Expected mean to be 0.5 after padding"
    assert image.dtype == torch.float16, "Expected dtype to be float16 after padding"

    with pytest.raises(Exception):
        pad_dim_to_size(image, 2, 1)
