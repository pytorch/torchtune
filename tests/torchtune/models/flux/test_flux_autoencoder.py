# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import mps_ignored_test

from torchtune.models.flux import flux_1_autoencoder
from torchtune.training.seed import set_seed

BSZ = 32
CH_IN = 3
RESOLUTION = 16
CH_MULTS = [1, 2]
CH_Z = 4
RES_Z = RESOLUTION // len(CH_MULTS)


@pytest.fixture(autouse=True)
def random():
    set_seed(0)


class TestFluxAutoencoder:
    @pytest.fixture
    def model(self):
        model = flux_1_autoencoder(
            resolution=RESOLUTION,
            ch_in=CH_IN,
            ch_out=3,
            ch_base=32,
            ch_mults=CH_MULTS,
            ch_z=CH_Z,
            n_layers_per_resample_block=2,
            scale_factor=1.0,
            shift_factor=0.0,
        )

        for param in model.parameters():
            param.data.uniform_(0, 0.1)

        return model

    @pytest.fixture
    def img(self):
        return torch.randn(BSZ, CH_IN, RESOLUTION, RESOLUTION)

    @pytest.fixture
    def z(self):
        return torch.randn(BSZ, CH_Z, RES_Z, RES_Z)

    @mps_ignored_test()
    def test_forward(self, model, img):
        actual = model(img)
        assert actual.shape == (BSZ, CH_IN, RESOLUTION, RESOLUTION)

        actual = torch.mean(actual, dim=(0, 2, 3))
        expected = torch.tensor([0.4286, 0.4276, 0.4054])
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_backward(self, model, img):
        y = model(img)
        loss = y.mean()
        loss.backward()

    @mps_ignored_test()
    def test_encode(self, model, img):
        actual = model.encode(img)
        assert actual.shape == (BSZ, CH_Z, RES_Z, RES_Z)

        actual = torch.mean(actual, dim=(0, 2, 3))
        expected = torch.tensor([0.6150, 0.7959, 0.7178, 0.7011])
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_decode(self, model, z):
        actual = model.decode(z)
        assert actual.shape == (BSZ, CH_IN, RESOLUTION, RESOLUTION)

        actual = torch.mean(actual, dim=(0, 2, 3))
        expected = torch.tensor([0.4246, 0.4241, 0.4014])
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
