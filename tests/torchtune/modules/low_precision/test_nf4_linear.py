# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from torchtune.modules.low_precision import FrozenNF4Linear
from torchtune.utils.seed import set_seed

try:
    from torchao.dtypes.nf4tensor import NF4Tensor
except ImportError as e:
    raise RuntimeError(
        "Please install torchao to run this test."
        "Example: pip install git+https://github.com/pytorch-labs/ao.git"
    ) from e

import bitsandbytes as bnb


@pytest.fixture(autouse=True)
def random():
    set_seed(31)


def _build_bnb_linear(input_weight):
    """
    Builds a bnb.nn.LinearNF4 from a given input weight
    """
    param = bnb.nn.Params4bit(input_weight, requires_grad=False, quant_type="nf4")
    bnb_linear = bnb.nn.LinearNF4(
        input_weight.size(0), input_weight.size(1), bias=False
    )
    bnb_linear.weight = param
    bnb_linear.cuda()
    return bnb_linear


class TestNF4Linear:
    """
    Class for testing our NF4Linear implementation.
    """

    def test_bias_unsupported(self):
        with pytest.raises(RuntimeError, match="does not currently support biases"):
            _ = FrozenNF4Linear(1, 1, bias=True)

    def test_non_bf16_unsupported(self):
        with pytest.raises(RuntimeError, match="only supported with bf16"):
            _ = FrozenNF4Linear(1, 1, dtype=torch.float32)

    def test_parameters(self):
        nf4_linear = FrozenNF4Linear(512, 512, device="cpu", dtype=torch.bfloat16)
        params = list(nf4_linear.parameters())
        assert len(params) == 1
        assert isinstance(params[0], NF4Tensor)

    def test_state_dict(self):
        nf4_linear = FrozenNF4Linear(512, 512, device="cpu", dtype=torch.bfloat16)
        state_dict = nf4_linear.state_dict()
        assert len(state_dict) == 1
        assert isinstance(state_dict["weight"], NF4Tensor)

    def test_frozen_nf4_linear(self):
        nf4_linear = FrozenNF4Linear(512, 512, device="cpu", dtype=torch.bfloat16)
        assert isinstance(nf4_linear.weight, NF4Tensor)
        assert torch.bfloat16 == nf4_linear.weight.get_original_weight().dtype

    def test_output_bf16(self):
        # Test to ensure W4 A16 produces A16
        nf4_linear = FrozenNF4Linear(512, 512, device="cpu", dtype=torch.bfloat16)
        inp = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        out = nf4_linear(inp)
        assert out.dtype == torch.bfloat16

    def test_backward_bf16(self):
        # Test to ensure backward pass gives activation a bf16 gradient and no gradient
        # to the linear's weight, as it is frozen.
        nf4_linear = FrozenNF4Linear(512, 512, device="cpu", dtype=torch.bfloat16)
        inp = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        nf4_linear(inp).sum().backward()
        assert inp.grad is not None and inp.grad.dtype == torch.bfloat16
        assert nf4_linear.weight.grad is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_nf4_reconstruction_vs_bnb(self):
        """
        Ensures a BNB NF4 linear and our FrozenNF4Linear have low error when
        reconstructing the respective original weights.
        """
        dim = 512
        nf4_linear = FrozenNF4Linear(dim, dim, device="cuda", dtype=torch.bfloat16)
        orig_weight = nf4_linear.weight.get_original_weight().clone().detach()
        bnb_nf4_linear = _build_bnb_linear(input_weight=orig_weight)

        # From https://github.com/drisspg/transformer_nuggets/blob/f05afad68ad9086d342268f46a7f344617a02314/test/test_qlora.py#L65
        bnb_reconstruction = bnb_nf4_linear(
            torch.eye(dim, dim, dtype=torch.bfloat16, device="cuda")
        )
        # Ensure nf4_linear and bnb reconstructions are close to each other.
        diff = (
            (bnb_reconstruction.T - nf4_linear.weight.get_original_weight()).abs().max()
        )
        assert diff.item() < 1e-2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_nf4_bnb_linear(self):
        """
        This test ensures that nf4_linear is "no worse" than BNB by ensuring the
        error compared to a bf16 linear is not more than BNB's implementation.
        """
        dim = 512
        nf4_linear = FrozenNF4Linear(dim, dim, device="cuda", dtype=torch.bfloat16)
        orig_weight = nf4_linear.weight.get_original_weight().clone().detach()
        bnb_nf4_linear = _build_bnb_linear(input_weight=orig_weight)
        bf16_linear = torch.nn.Linear(dim, dim, device="cuda", dtype=torch.bfloat16)

        inp = torch.randn(2, 512, dtype=torch.bfloat16, device="cuda")

        out_nf4 = nf4_linear(inp)
        out_bnb = bnb_nf4_linear(inp)
        out_ref = bf16_linear(inp)

        err_bnb = (out_bnb - out_ref).sum().abs().max()
        err_native = (out_nf4 - out_ref).sum().abs().max()
        assert err_native.item() <= err_bnb
