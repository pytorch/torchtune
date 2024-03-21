# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
from pathlib import Path

import torchtune

root = Path(torchtune.__file__).parent.parent.absolute()
CACHE_ARTIFACTS_SCRIPT_PATH = Path.joinpath(
    root, "tests", "regression_tests", "cache_artifacts.sh"
)


def pytest_sessionstart(session):
    subprocess.call(["sh", CACHE_ARTIFACTS_SCRIPT_PATH])
