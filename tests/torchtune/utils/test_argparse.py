#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import mock
from unittest.mock import mock_open

import pytest

from torchtune.utils import ArgumentParser

config = """
         a: 1
         b: 2
         """


class TestArgParse:
    @pytest.fixture
    def parser(self):
        parser = ArgumentParser("Test parser")
        parser.add_argument("--a", type=int, default=0)
        parser.add_argument("--b", type=int, default=0)
        parser.add_argument("--c", type=int, default=0)
        return parser

    @mock.patch("builtins.open", mock_open(read_data=config))
    def test_parse_args(self, parser):
        args = parser.parse_args(["--config", "test.yaml", "--b", "3"])
        assert args.a == 1, f"a == {args.a} not 1 as set in the config."
        assert args.b == 3, f"b == {args.b} not 3 as set in the command args."
        assert args.c == 0, f"c == {args.c} not 0 as set in the argument default."

    @mock.patch("builtins.open", mock_open(read_data="d: 4"))
    def test_read_bad_config(self, parser):
        with pytest.raises(AssertionError):
            parser.parse_args(["--config", "test.yaml"])

    def test_required_argument(self, parser):
        with pytest.raises(AssertionError):
            parser.add_argument("--d", required=True, type=int, default=0)
