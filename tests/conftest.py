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
    of remote artifacts needed for the test run and filtering across unit tests, recipe tests, and
    regression tests.

    When testing, you should run one of the following:

    - `pytest tests`: run unit tests only
    - `pytest tests --with-integration`: run unit tests and recipe tests
    - `pytest tests --with-integration --with-slow-integration`: run all tests
    - `pytest tests -m integration_test`: run recipe tests only
    - `pytest tests -m slow_integration_test`: run regression tests only

    Similar commands apply for filtering in subdirectories or individual test files.

    This hook also ensures that the appropriate artifacts are available locally for all of the above cases.
    Note that artifact download is determined by the CLI flags, so if you run e.g.
    `pytest tests/torchtune/some_unit_test.py -m integration_test`, the integration test
    artifacts will be downloaded even if your test doesn't require them.

    The hook also supports optional silencing of S3 progress bars to reduce CI log spew via `--silence-s3-logs`.
    """

    # To make it more convenient to run an individual unit test, we override the default
    # behavior of pytest-integration to run with --without-integration --without-slow-integration
    # This means that we need to manually override the values of run_integration and run_slow_integration
    # whenever either set of tests is passed via the -m option.

    if config.option.markexpr == "integration_test":
        config.option.run_integration = True
        run_regression_tests = False
    if config.option.markexpr == "slow_integration_test":
        config.option.run_slow_integration = True
        run_recipe_tests = False

    # Default is to run both integration and slow integration tests (i.e. both are None)
    run_recipe_tests = (
        config.option.run_integration is None or config.option.run_integration is True
    )
    run_regression_tests = (
        config.option.run_slow_integration is None
        or config.option.run_slow_integration is True
    )

    cmd = str(CACHE_ARTIFACTS_SCRIPT_PATH)

    if run_recipe_tests:
        cmd += " --run-recipe-tests"
    if run_regression_tests:
        cmd += " --run-regression-tests"

    # Optionally silence S3 download logs (useful when running on CI)
    if config.option.silence_s3_logs:
        cmd += " --silence-s3-logs"

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
    parser.addoption(
        "--silence-s3-logs",
        action="store_true",
        help="Silence progress bar when fetching assets from S3",
    )
