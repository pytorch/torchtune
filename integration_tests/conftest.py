# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest


def pytest_addoption(parser: argparse.ArgumentParser) -> None:
    parser.addoption(
        "--llama2-path",
        action="store",
        type=str,
        help="Path to the llama2-7b model checkpoint",
    )
    parser.addoption(
        "--tokenizer-path",
        action="store",
        type=str,
        help="Path to the tokenizer checkpoint",
    )


@pytest.fixture
def llama2_path(request):
    return request.config.getoption("--llama2-path")


@pytest.fixture
def tokenizer_path(request):
    return request.config.getoption("--tokenizer-path")
