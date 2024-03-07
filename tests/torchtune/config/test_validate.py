# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from omegaconf import OmegaConf
from torchtune import config


class TestValidate:
    def valid_config_string(self):
        return """
        test:
          _component_: torchtune.utils.get_dtype
          dtype: fp32
        """

    def invalid_config_string(self):
        return """
        test:
          _component_: torchtune.utils.get_dtype
          dtype: fp32
          dummy: 3
        """

    def test_validate(self):
        conf = OmegaConf.create(self.valid_config_string())
        # Test a valid component
        config.validate(conf)
        # Test an invalid component
        conf = OmegaConf.create(self.invalid_config_string())
        with pytest.raises(
            TypeError, match="got an unexpected keyword argument 'dummy'"
        ):
            config.validate(conf)
