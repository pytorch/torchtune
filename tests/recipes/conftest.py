# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import torchtune

root = Path(torchtune.__file__).parent.parent.absolute()
CACHE_ARTIFACTS_SCRIPT_PATH = Path.joinpath(
    root, "tests", "recipes", "cache_artifacts.sh"
)


def pytest_sessionstart(session):
    os.system(CACHE_ARTIFACTS_SCRIPT_PATH)
