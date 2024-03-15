# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from omegaconf import OmegaConf
from torchtune import config
from torchtune.config._errors import ConfigError

VALID_CONFIG_PATH = "tests/assets/valid_dummy_config.yaml"
INVALID_CONFIG_PATH = "tests/assets/invalid_dummy_config.yaml"


class TestValidate:
    def test_validate(self):
        conf = OmegaConf.load(VALID_CONFIG_PATH)
        # Test a valid component
        config.validate(conf)
        # Test an invalid component
        conf = OmegaConf.load(INVALID_CONFIG_PATH)
        with pytest.raises(ConfigError) as excinfo:
            config.validate(conf)
        exc_config = excinfo.value
        assert len(exc_config.errors) == 2
        for e in exc_config.errors:
            assert isinstance(e, TypeError)
            assert str(e) == "get_dtype got an unexpected keyword argument 'dummy'"
