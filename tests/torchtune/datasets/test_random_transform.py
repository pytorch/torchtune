# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess

import pytest


class TestRandomTransformsWithReproducibleDataLoader:
    @pytest.mark.parametrize("rank", [0, 1])
    def test_multiworker_rank(self, rank):
        p = subprocess.run(
            [
                "python3",
                "tests/torchtune/datasets/sample_dataset.py",
                "--seed",
                "11",
                "--rank",
                str(rank),
                "--num_workers",
                "2",
            ],
            capture_output=True,
            text=True,
        )
        output_ids = set()
        for line in p.stdout.splitlines():
            output_ids.add(int(line))
        if rank == 0:
            expected_output_ids = set([3325388913236545215, 3325388913236545216])
        else:
            expected_output_ids = set([2843802486886340251, 2843802486886340252])
        assert expected_output_ids == output_ids

    def test_no_worker(self):
        p = subprocess.run(
            [
                "python3",
                "tests/torchtune/datasets/sample_dataset.py",
                "--seed",
                "13",
                "--rank",
                "0",
                "--num_workers",
                "0",
            ],
            capture_output=True,
            text=True,
        )
        output_ids = set()
        for line in p.stdout.splitlines():
            output_ids.add(int(line))
        expected_output_ids = set([13])
        assert expected_output_ids == output_ids
