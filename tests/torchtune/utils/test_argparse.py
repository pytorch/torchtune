#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import mock

import pytest

from torchtune.utils import TuneArgumentParser

_CONFIG = {"a": 1, "b": 2}


class TestArgParse:
    @pytest.fixture
    def parser(self):
        parser = TuneArgumentParser("Test parser")
        return parser

    @mock.patch("torchtune.utils.argparse.OmegaConf.load", return_value=_CONFIG)
    def test_parse_args(self, mock_load, parser):
        """
        Test that the parser can load a config and override parameters provided on CLI.
        The actual load is mocked to return the test config above.
        """
        args = parser.parse_args(["--config", "test.yaml", "--override", "b=3", "c=4"])
        assert args.a == 1, f"a == {args.a} not 1 as set in the config."
        assert args.b == 3, f"b == {args.b} not 3 as set in the command args."
        assert args.c == 4, f"c == {args.c} not 4 as set in the command args."
        assert len(vars(args).keys() - {"a", "b", "c"}) == 0, "Extra args found."

    def test_required_argument(self, parser):
        """
        Test that the parser does not allow required arguments to be added
        """
        with pytest.raises(AssertionError):
            parser.add_argument("--d", required=True, type=int, default=0)
