# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import uuid

import pytest
import torch.distributed.launcher as pet


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
