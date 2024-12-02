# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Tuple

import pytest
import torch
from tests.test_utils import assert_expected
from torchtune.modules.layer_dropout import (
    get_scale,
    LayerDropout,
    ModuleLayerDropoutWrapper,
    prepare_layer_dropout,
    ScaleType,
)


class TestLayerDropout:
    """Class for testing LayerDropout implementation."""

    @pytest.fixture
    def input_shape(self) -> Tuple[int, int]:
        bsz = 8
        seqlen = 256
        dim = 32
        return bsz, seqlen, dim

    @pytest.fixture
    def input(self, input_shape: Tuple[int]) -> torch.Tensor:
        return torch.randn(input_shape)

    @pytest.fixture
    def layer_dropout(
        self, prob: float = 0.5, disable_on_eval: bool = True
    ) -> LayerDropout:
        return LayerDropout(prob=prob, disable_on_eval=disable_on_eval)

    def test_forward_train_prob_1(
        self, layer_dropout: LayerDropout, input: torch.Tensor
    ):
        # With dropout probability = 1.0, we expect output to be the same as input
        layer_dropout.prob = 1.0
        output = layer_dropout.forward(lambda x: x**2, input)
        assert torch.allclose(output, input)

    def test_forward_train_prob_0(
        self, layer_dropout: LayerDropout, input: torch.Tensor
    ):
        # With dropout probability = 1.0, we expect the operation to be applied on all elements in the input
        layer_dropout.prob = 0.0
        output = layer_dropout.forward(lambda x: x**2, input)
        assert torch.allclose(output, input**2)

    def test_forward_eval(self, layer_dropout: LayerDropout, input: torch.Tensor):
        layer_dropout.prob = 1.0
        layer_dropout.eval()

        layer_dropout.disable_on_eval = True
        output = layer_dropout.forward(lambda x: x**2, input)
        assert torch.allclose(output, input**2)

        layer_dropout.disable_on_eval = False
        with torch.no_grad():
            output = layer_dropout.forward(lambda x: x**2, input)
        assert torch.allclose(output, input)


class TestLayerDropoutWrapper:
    @pytest.fixture
    def input_shape(self) -> Tuple[int, int]:
        bsz = 4
        dim = 8
        return (bsz, dim)

    @pytest.fixture
    def input(self, input_shape: Tuple[int]) -> torch.Tensor:
        return torch.randn(input_shape)

    @pytest.fixture
    def model(self, input_shape) -> torch.nn.Module:
        _, dim = input_shape
        return torch.nn.Sequential(
            torch.nn.Linear(dim, 32), torch.nn.ReLU(), torch.nn.Linear(32, dim)
        )

    @pytest.fixture
    def linear(self, input_shape) -> torch.nn.Module:
        _, dim = input_shape
        return torch.nn.Linear(dim, dim)

    def test_linear(self, linear: torch.nn.Module, input: torch.Tensor):
        wrapper = ModuleLayerDropoutWrapper(linear, LayerDropout(prob=0.5))
        assert wrapper.module == linear

        # Test output
        wrapper.dropout.prob = 1
        assert torch.allclose(wrapper(input), input)
        wrapper.dropout.prob = 0
        assert torch.allclose(wrapper(input), linear(input))

        # Test getters
        assert wrapper.in_features == linear.in_features
        assert wrapper.out_features == linear.out_features
        assert torch.equal(wrapper.weight, linear.weight)

        # Test setters
        wrapper.weight.data = wrapper.weight.data * 2
        assert torch.equal(wrapper.weight, linear.weight)

        # Test state_dict
        for k in wrapper.state_dict().keys():
            assert torch.equal(wrapper.state_dict()[k], linear.state_dict()[k])

    def test_model(self, model: torch.nn.Module, input: torch.Tensor):
        wrapper = ModuleLayerDropoutWrapper(model, LayerDropout(prob=0.5))
        assert wrapper.module == model

        # Test output
        wrapper.dropout.prob = 1
        assert torch.allclose(wrapper(input), input)
        wrapper.dropout.prob = 0
        assert torch.allclose(wrapper(input), model(input))

        # Test getters
        assert wrapper[0].in_features == model[0].in_features
        assert wrapper[0].out_features == model[0].out_features
        assert torch.equal(wrapper[0].weight, model[0].weight)

        # Test setters
        wrapper[2].weight.data = wrapper[2].weight.data * 2
        assert torch.equal(wrapper[2].weight, model[2].weight)

        # Test state_dict
        for k in wrapper.state_dict().keys():
            assert torch.equal(wrapper.state_dict()[k], model.state_dict()[k])


class TestScales:
    def test_get_scale_uniform(self):
        scale_type = ScaleType.UNIFORM
        scale_period = 10

        assert_expected(get_scale(scale_type, scale_period, 0), 1.0)
        assert_expected(get_scale(scale_type, scale_period, scale_period / 2), 1.0)
        assert_expected(get_scale(scale_type, scale_period, scale_period), 1.0)
        assert_expected(get_scale(scale_type, scale_period, scale_period * 2), 1.0)

    def test_get_scale_linear(self):
        scale_type = ScaleType.LINEAR
        scale_period = 10

        assert_expected(get_scale(scale_type, scale_period, 0), 0.0)
        assert_expected(get_scale(scale_type, scale_period, scale_period / 2), 1 / 2)
        assert_expected(get_scale(scale_type, scale_period, scale_period), 1.0)
        assert_expected(get_scale(scale_type, scale_period, scale_period * 2), 1.0)

    def test_get_scale_exp(self):
        scale_type = ScaleType.EXP
        scale_period = 10

        assert_expected(get_scale(scale_type, scale_period, 0), 0.0)
        assert_expected(
            get_scale(scale_type, scale_period, scale_period / 2),
            math.pow(2, 1 / 2) - 1,
        )
        assert_expected(get_scale(scale_type, scale_period, scale_period), 1.0)
        assert_expected(get_scale(scale_type, scale_period, scale_period * 2), 1.0)

    def test_get_scale_log(self):
        scale_type = ScaleType.LOG
        scale_period = 10

        assert_expected(get_scale(scale_type, scale_period, 0), 0.0)
        assert_expected(
            get_scale(scale_type, scale_period, scale_period / 2),
            math.log(5 + 1, scale_period + 1),
        )
        assert_expected(get_scale(scale_type, scale_period, scale_period), 1.0)
        assert_expected(get_scale(scale_type, scale_period, scale_period * 2), 1.0)

    def test_get_scale_sin(self):
        scale_type = ScaleType.SIN
        scale_period = 10

        assert_expected(get_scale(scale_type, scale_period, 0), 0.0)
        assert_expected(
            get_scale(scale_type, scale_period, scale_period / 2),
            math.sin(0.5 * math.pi * 0.5),
        )
        assert_expected(get_scale(scale_type, scale_period, scale_period), 1.0)
        assert_expected(get_scale(scale_type, scale_period, scale_period * 2), 1.0)

    def test_get_scale_sigmoid(self):
        scale_type = ScaleType.SIGMOID
        scale_period = 10

        # sigmoid(0) is close to 0 but not 0, hence adding relatively large rotl and atol
        assert_expected(
            get_scale(scale_type, scale_period, 0), 0.0, rtol=1e-2, atol=1e-2
        )
        assert_expected(
            get_scale(scale_type, scale_period, scale_period / 2),
            0.5,
        )
        assert_expected(get_scale(scale_type, scale_period, scale_period), 1.0)
        assert_expected(get_scale(scale_type, scale_period, scale_period * 2), 1.0)


class TestLayerDropoutModel:
    def test_prepare_layer_dropout_uniform(self):
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(10, 10) for _ in range(5)]
                )

        model = MockModel()
        prob_max = 0.5
        prob_layer_scale = ScaleType.UNIFORM
        layers_str = "0:4"
        prepare_layer_dropout(model.layers, prob_max, prob_layer_scale, layers_str)
        for i, layer in enumerate(model.layers):
            assert hasattr(layer, "dropout")
            if i in range(0, 4):
                assert layer.dropout.prob == prob_max
            else:
                assert layer.dropout.prob == 0

    def test_prepare_layer_dropout_exp(self):
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(10, 10) for _ in range(5)]
                )

        model = MockModel()
        prob_max = 0.5
        prob_layer_scale = ScaleType.EXP
        layers_str = ":"
        prepare_layer_dropout(model.layers, prob_max, prob_layer_scale, layers_str)
        for i, layer in enumerate(model.layers):
            assert hasattr(layer, "dropout")
            if i == 0:
                assert layer.dropout.prob == 0
            elif i == len(model.layers) - 1:
                assert layer.dropout.prob == prob_max
            else:
                assert layer.dropout.prob > 0 and layer.dropout.prob < prob_max

    def test_prepare_layer_dropout_linear(self):
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(10, 10) for _ in range(5)]
                )

        model = MockModel()
        prob_max = 0.5
        prob_layer_scale = ScaleType.LINEAR
        layers_str = ":"
        prepare_layer_dropout(model.layers, prob_max, prob_layer_scale, layers_str)
        for i, layer in enumerate(model.layers):
            assert hasattr(layer, "dropout")
            if i == 0:
                assert layer.dropout.prob == 0
            elif i == len(model.layers) - 1:
                assert layer.dropout.prob == prob_max
            elif i == len(model.layers) / 2:
                assert layer.dropout.prob == prob_max / 2
            else:
                assert layer.dropout.prob >= 0.0 and layer.dropout.prob <= prob_max
