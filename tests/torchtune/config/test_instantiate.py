# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from textwrap import dedent

import pytest
from omegaconf import OmegaConf
from torchtune.config._errors import InstantiationError
from torchtune.config._instantiate import (
    _create_component,
    _instantiate_node,
    instantiate,
)
from torchtune.config._utils import _has_component
from torchtune.modules import RMSNorm


class TestInstantiate:
    @pytest.fixture
    def config(self):
        s = """
        a: b
        b: c
        test:
          _component_: torchtune.modules.RMSNorm
          dim: 5
        """
        return OmegaConf.create(s)

    @pytest.fixture
    def module(self):
        return RMSNorm(dim=5, eps=1e-4)

    def get_dim(self, rms_norm: RMSNorm):
        return rms_norm.scale.shape[0]

    def test_has_path(self, config):
        assert _has_component(config.test)
        assert not _has_component(config.a)

    def test_call_object(self, module):
        obj = RMSNorm
        args = (5,)
        kwargs = {"eps": 1e-4}
        actual = _create_component(obj, args, kwargs)
        expected = module
        assert isinstance(actual, RMSNorm)
        assert self.get_dim(actual) == self.get_dim(expected)
        assert actual.eps == expected.eps

    def test_instantiate_node(self, config, module):
        actual = _instantiate_node(config.test)
        expected = module
        assert isinstance(actual, RMSNorm)
        assert self.get_dim(actual) == self.get_dim(expected)

        with pytest.raises(
            InstantiationError, match="Cannot instantiate specified object"
        ):
            _ = _instantiate_node(config.a)

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

    def test_tokenizer_config_with_null(self):
        assets = Path(__file__).parent.parent.parent / "assets"
        s = dedent(
            f"""\
        tokenizer:
          _component_: torchtune.models.llama2.llama2_tokenizer
          max_seq_len: null
          path: {assets / 'm.model'}
        """
        )
        config = OmegaConf.create(s)

        tokenizer = instantiate(config.tokenizer)
        assert tokenizer.max_seq_len is None
