# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.models.llama2 import scale_hidden_dim_for_mlp


def _get_expected_scaled_dim(input_dim, mul_of, dim_multiplier=None):
    expected = 4 * int(2 * input_dim / 3)
    if dim_multiplier:
        expected = int(expected * dim_multiplier)
    expected = mul_of * ((expected + mul_of - 1) // mul_of)
    return expected


def test_scale_hidden_dim_for_mlp():
    # Test that the hidden dim is scaled correctly.
    dim = 256
    mul_of = 1024
    dim_multiplier = 1.3

    expected = _get_expected_scaled_dim(dim, mul_of, dim_multiplier)
    assert scale_hidden_dim_for_mlp(dim, mul_of, dim_multiplier) == expected

    expected_no_multiplier = _get_expected_scaled_dim(dim, mul_of, dim_multiplier=None)
    assert scale_hidden_dim_for_mlp(dim, mul_of) == expected_no_multiplier
