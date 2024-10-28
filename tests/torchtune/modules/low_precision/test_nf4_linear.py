# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

try:
    import bitsandbytes as bnb

    bnb_installed = True
except ImportError:
    bnb_installed = False
import pytest
import torch
from torchao.dtypes.nf4tensor import NF4Tensor
from torchtune.modules.low_precision import FrozenNF4Linear
from torchtune.training.seed import set_seed


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

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_parameters(self, dtype):
        nf4_linear = FrozenNF4Linear(512, 512, device="cpu", dtype=dtype)
        params = list(nf4_linear.parameters())
        assert len(params) == 1
        assert isinstance(params[0], NF4Tensor)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_state_dict(self, dtype):
        nf4_linear = FrozenNF4Linear(512, 512, device="cpu", dtype=dtype)
        state_dict = nf4_linear.state_dict()
        assert len(state_dict) == 1
        assert isinstance(state_dict["weight"], NF4Tensor)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("bias", [True, False])
    def test_output_dtype(self, dtype, bias):
        # Test to ensure W4 A16 produces A16 / W4A32 produces A32
        nf4_linear = FrozenNF4Linear(512, 512, device="cpu", dtype=dtype, bias=bias)
        inp = torch.randn(2, 512, dtype=dtype, requires_grad=True)
        out = nf4_linear(inp)
        assert out.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_backward_dtype(self, dtype):
        # Test to ensure backward pass gives activation a bf16 gradient and no gradient
        # to the linear's weight, as it is frozen.
        nf4_linear = FrozenNF4Linear(512, 512, device="cpu", dtype=dtype)
        inp = torch.randn(2, 512, dtype=dtype, requires_grad=True)
        nf4_linear(inp).sum().backward()
        assert inp.grad is not None and inp.grad.dtype == dtype
        assert nf4_linear.weight.grad is None

    @pytest.mark.skipif(not bnb_installed, reason="bitsandbytes is not installed")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_nf4_reconstruction_vs_bnb(self, dtype):
        """
        Ensures a BNB NF4 linear and our FrozenNF4Linear have low error when
        reconstructing the respective original weights.
        """
        dim = 512
        nf4_linear = FrozenNF4Linear(dim, dim, device="cuda", dtype=dtype)
        orig_weight = nf4_linear.weight.get_original_weight().clone().detach()
        bnb_nf4_linear = _build_bnb_linear(input_weight=orig_weight)

        # From https://github.com/drisspg/transformer_nuggets/blob/f05afad68ad9086d342268f46a7f344617a02314/test/test_qlora.py#L65
        bnb_reconstruction = bnb_nf4_linear(
            torch.eye(dim, dim, dtype=dtype, device="cuda")
        )
        # Ensure nf4_linear and bnb reconstructions are close to each other.
        assert torch.allclose(
            bnb_reconstruction.T, nf4_linear.weight.get_original_weight(), 1e-2
        )

    @pytest.mark.skipif(not bnb_installed, reason="bitsandbytes is not installed")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_nf4_bnb_linear(self, dtype):
        """
        This test ensures that nf4_linear is "no worse" than BNB by ensuring the
        error compared to a bf16 linear is not more than BNB's implementation.
        """
        dim = 512
        nf4_linear = FrozenNF4Linear(dim, dim, device="cuda", dtype=dtype)
        orig_weight = nf4_linear.weight.get_original_weight().clone().detach()
        bnb_nf4_linear = _build_bnb_linear(input_weight=orig_weight)
        bf16_linear = torch.nn.Linear(dim, dim, device="cuda", dtype=dtype)

        inp = torch.randn(2, 512, dtype=dtype, device="cuda")

        out_nf4 = nf4_linear(inp)
        out_bnb = bnb_nf4_linear(inp)
        out_ref = bf16_linear(inp)

        err_bnb = out_bnb - out_ref
        err_native = out_nf4 - out_ref
        assert torch.allclose(err_bnb, err_native, 1.0e-2, 1.0e-2)
