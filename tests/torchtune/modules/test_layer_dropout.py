# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple
import pytest
import torch
from tests.test_utils import assert_expected
from torchtune.modules.layer_dropout import LayerDropout


class TestLayerDropout:
    """Class for testing LayerDropout implementation."""


    @pytest.fixture(autouse=True)
    def random(self):
        torch.manual_seed(0)


    @pytest.fixture
    def input_shape(self) -> Tuple[int, int]:
        bsz = 32
        seqlen = 1024
        dim = 4096
        return bsz, seqlen, dim


    @pytest.fixture
    def input(self, input_shape: Tuple[int]) -> torch.Tensor:
        return torch.randn(input_shape)


    @pytest.fixture
    def layer_dropout(self, prob: float = 0.5, disable_on_eval: bool = True) -> LayerDropout:
        return LayerDropout(prob=prob, disable_on_eval=disable_on_eval)


    def test_forward_train_prob_1(self, layer_dropout: LayerDropout, input: torch.Tensor) -> None:
        # With dropout probability = 1.0, we expect output to be the same as input
        layer_dropout.prob = 1.0
        output = layer_dropout.forward(lambda x: x**2, input)
        assert torch.allclose(output, input, atol=1e-7, rtol=1e-3)


    def test_forward_train_prob_0(self, layer_dropout: LayerDropout, input: torch.Tensor) -> None:
        # With dropout probability = 1.0, we expect the operation to be applied on all elements in the input
        layer_dropout.prob = 0.0
        output = layer_dropout.forward(lambda x: x**2, input)
        assert torch.allclose(output, input**2, atol=1e-7, rtol=1e-3)


    def test_forward_eval(self, layer_dropout: LayerDropout, input: torch.Tensor) -> None:
        layer_dropout.prob = 1.0
        layer_dropout.eval()

        layer_dropout.disable_on_eval = True
        output = layer_dropout.forward(lambda x: x**2, input)
        assert torch.allclose(output, input**2, atol=1e-7, rtol=1e-3)

        layer_dropout.disable_on_eval = False
        with torch.no_grad():
            output = layer_dropout.forward(lambda x: x**2, input)
        assert torch.allclose(output, input, atol=1e-7, rtol=1e-3)