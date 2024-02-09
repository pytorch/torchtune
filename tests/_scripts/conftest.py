# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse


def pytest_addoption(parser: argparse.ArgumentParser) -> None:
    parser.addoption(
        "--large-scale",
        type=bool,
        default=False,
        help="Run a larger scale integration test",
    )
