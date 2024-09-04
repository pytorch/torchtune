# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf
from torchtune import config

from torchtune.config._parse import TuneRecipeArgumentParser

_CONFIG = {"a": 1, "b": 2}


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
            return_value=(Namespace(**_CONFIG), []),
        ) as mock_parse_args:
            with pytest.raises(SystemExit):
                func()
            mock_parse_args.assert_called_once()


class TestArgParse:
    @pytest.fixture
    def parser(self):
        parser = TuneRecipeArgumentParser("Test parser")
        return parser

    @patch("torchtune.config._parse.OmegaConf.load", return_value=_CONFIG)
    def test_parse_known_args(self, mock_load, parser):
        """
        Test that the parser can load a config and override parameters provided on CLI.
        The actual load is mocked to return the test config above.
        """
        config_args, cli_args = parser.parse_known_args(
            ["--config", "test.yaml", "b=3", "c=4"]
        )
        assert config_args.a == 1, f"a == {config_args.a} not 1 as set in the config."
        assert config_args.b == 2, f"b == {config_args.b} not 2 as set in the config."

        cli_kwargs = OmegaConf.from_dotlist(cli_args)
        assert (
            cli_kwargs.b == 3
        ), f"b == {cli_kwargs.b} not 3 as set in the command args."
        assert (
            cli_kwargs.c == 4
        ), f"c == {cli_kwargs.c} not 4 as set in the command args."

        with pytest.raises(ValueError, match="Additional flag arguments not supported"):
            _ = parser.parse_known_args(
                ["--config", "test.yaml", "--b", "3"],
            )
