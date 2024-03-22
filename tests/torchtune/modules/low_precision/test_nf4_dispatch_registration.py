# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchao.dtypes import to_nf4


class TestNF4DispatchRegistration:
    """
    Class for testing NF4Tensor dispatch ops.
    """

    def test_inplace_copy_copies_expected_attributes(self):
        """
        This test ensures that we're copying over all relevant attributes when implementing
        torch.ops.aten.copy_.default. If this test fails, we would need to update our implementation
        in _register_nf4_dispatch_ops to cover the newly added attributes.
        """
        expected_inplace_copy_attrs = [
            "block_size",
            "n_blocks",
            "scaler_block_size",
            "quantized_scalers",
            "quantization_factor",
            "scaler_mean",
            "quantized_data",
            "nf4",
        ]

        z = to_nf4(torch.rand(512, 512, dtype=torch.bfloat16))
        inplace_copy_attr_set = set(z.__dict__.keys())
        assert set(expected_inplace_copy_attrs) == inplace_copy_attr_set
