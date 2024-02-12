# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest

import torch

from tests.test_utils import assert_expected

from torchtune.utils.logits_transforms import (
    TemperatureTransform,
    TopKTransform,
    TopPTransform,
)


class TestTemperatureTransform:
    def test_invalid_temperature(self):
        """
        Test if ValueError is raised when temperature is set to 0.0
        """
        with pytest.raises(ValueError):
            _ = TemperatureTransform(temperature=0.0)

    def test_default(self):
        """
        Test if the TemperatureTransform correctly transforms the logits
        """
        temperature = 0.6
        transform = TemperatureTransform(temperature=temperature)
        logits = torch.arange(0, 1, 0.1)[None]

        expected = logits / temperature
        actual = transform(logits)
        assert_expected(actual, expected)


class TestTopPTransform:
    def test_invalid_p(self):
        """
        Test if ValueError is raised when prob is set to -1.0, 0.0, or 2.0
        """
        for prob in (-1.0, 0.0, 2.0):
            with pytest.raises(ValueError):
                _ = TopPTransform(prob=prob)

    def test_default(self):
        """
        Test if the TopPTransform correctly transforms the logits
        """
        prob = 0.5
        transform = TopPTransform(prob=prob)
        logits = torch.arange(0.1, 0.5, 0.1)[None]

        expected = torch.tensor([[0.0, 0.0, 0.3, 0.4]]) / 0.7
        actual = transform(logits)
        assert_expected(actual, expected)


class TestTopKTransform:
    def test_invalid_k(self):
        """
        Test if ValueError is raised when top_k is set to -1 or TypeError is raised when top_k is set to 0.5
        """
        with pytest.raises(ValueError):
            _ = TopKTransform(top_k=-1)

        with pytest.raises(TypeError):
            _ = TopKTransform(top_k=0.5)

    def test_default(self):
        """
        Test if the TopKTransform correctly transforms the logits
        """
        top_k = 3
        transform = TopKTransform(top_k=top_k)
        logits = torch.arange(0, 0.5, 0.1)[None]

        expected = torch.tensor([[0.0, 0.0, 0.2, 0.3, 0.4]]) / 0.9
        actual = transform(logits)
        assert_expected(actual, expected)
