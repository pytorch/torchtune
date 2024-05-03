# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from unittest.mock import patch

import pytest
from torchtune import config

_CONFIG = Namespace(a=1, b=2)


class TestParse:
    def test_parse(self):
        a = 1
        b = 3

        @config.parse
        def func(cfg):
            assert cfg.a == a
            assert cfg.b != b

        with patch(
            "torchtune.config._parse.TuneRecipeArgumentParser.parse_known_args",
            return_value=(_CONFIG, []),
        ) as mock_parse_args:
            with pytest.raises(SystemExit):
                func()
            mock_parse_args.assert_called_once()
