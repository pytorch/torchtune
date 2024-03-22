# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import uuid
from pathlib import Path

import pytest
import torch.distributed.launcher as pet

import torchtune

root = str(Path(torchtune.__file__).parent.parent.absolute())
CACHE_ARTIFACTS_SCRIPT_PATH = root + "/tests/cache_artifacts.sh"


def pytest_configure(config):
    """
    This hook runs before each pytest invocation. Its purpose is to handle optional fetching
    of remote artifacts needed for the test run. For testing, you should run one of the following:

    - `pytest tests --without-integration --without-slow-integration`: run unit tests only
    - `pytest tests --without-slow-integration`: run unit tests and recipe tests
    - `pytest tests`: run all tests
    - `pytest tests -m integration_test`: run recipe tests only
    - `pytest tests -m slow_integration_test`: run regression tests only

    This hook ensures that the appropriate artifacts are available locally for each of these cases.
    """
    # Default is to run both integration and slow integration tests (i.e. both are None)
    run_recipe_tests = (
        config.option.run_integration is None or config.option.run_integration is True
    )
    run_regression_tests = (
        config.option.run_slow_integration is None
        or config.option.run_slow_integration is True
    )

    # For -m flags, we run only those tests and so disable the others here
    if config.option.markexpr == "integration_test":
        run_regression_tests = False
    if config.option.markexpr == "slow_integration_test":
        run_recipe_tests = False

    cmd = str(CACHE_ARTIFACTS_SCRIPT_PATH)

    if run_recipe_tests:
        cmd += " --run-recipe-tests"
    if run_regression_tests:
        cmd += " --run-regression-tests"

    # Only need to handle artifacts for recipe and regression tests
    if run_recipe_tests or run_regression_tests:
        os.system(cmd)


@pytest.fixture(scope="session")
def get_pet_launch_config():
    def get_pet_launch_config_fn(nproc: int) -> pet.LaunchConfig:
        """
        Initialize pet.LaunchConfig for single-node, multi-rank functions.

        Args:
            nproc (int): The number of processes to launch.

        Returns:
            An instance of pet.LaunchConfig for single-node, multi-rank functions.

        Example:
            >>> from torch.distributed import launcher
            >>> launch_config = get_pet_launch_config(nproc=8)
            >>> launcher.elastic_launch(config=launch_config, entrypoint=train)()
        """
        return pet.LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=nproc,
            run_id=str(uuid.uuid4()),
            rdzv_backend="c10d",
            rdzv_endpoint="localhost:0",
            max_restarts=0,
            monitor_interval=1,
        )

    return get_pet_launch_config_fn


def pytest_addoption(parser: argparse.ArgumentParser) -> None:
    parser.addoption(
        "--large-scale",
        type=bool,
        default=False,
        help="Run a larger scale integration test",
    )
