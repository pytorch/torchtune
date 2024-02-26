# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from omegaconf import OmegaConf
from torchtune.config._instantiate import (
    call_object,
    has_path,
    instantiate,
    instantiate_node,
)
from torchtune.modules import RMSNorm


class TestInstantiate:
    @pytest.fixture
    def config(self):
        s = """
        a: b
        b: c
        test:
          _path_: torchtune.modules.RMSNorm
          dim: 5
        """
        return OmegaConf.create(s)

    @pytest.fixture
    def module(self):
        return RMSNorm(dim=5, eps=1e-4)

    def get_dim(self, rms_norm: RMSNorm):
        return rms_norm.scale.shape[0]

    def test_has_path(self, config):
        assert has_path(config.test)
        assert not has_path(config.a)

    def test_call_object(self, module):
        obj = RMSNorm
        args = (5,)
        kwargs = {"eps": 1e-4}
        actual = call_object(obj, args, kwargs)
        expected = module
        assert isinstance(actual, RMSNorm)
        assert self.get_dim(actual) == self.get_dim(expected)
        assert actual.eps == expected.eps

    def test_instantiate_node(self, config, module):
        actual = instantiate_node(config.test)
        expected = module
        assert isinstance(actual, RMSNorm)
        assert self.get_dim(actual) == self.get_dim(expected)

        with pytest.raises(ValueError, match="Cannot instantiate specified object"):
            _ = instantiate_node(config.a)

    def test_instantiate(self, config, module):
        actual = instantiate(config.test)
        expected = module
        assert isinstance(actual, RMSNorm)
        assert self.get_dim(actual) == self.get_dim(expected)

        # Test passing in kwargs
        actual = instantiate(config.test, eps=1e-4)
        assert actual.eps == expected.eps

        # Test passing in positional args
        del config.test.dim
        actual = instantiate(config.test, 3)
        assert self.get_dim(actual) == 3
