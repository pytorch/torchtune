# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os

import pytest

from recipes.params import FullFinetuneParams
from torchtune.utils.argparse import TuneArgumentParser

ROOT_DIR: str = os.path.join(os.path.abspath(__file__), "../../../configs")

config_to_params = {
    os.path.join(ROOT_DIR, "alpaca_llama2_full_finetune.yaml"): FullFinetuneParams,
}


class TestConfigs:
    """Tests that all configs are well formed.
    Configs should have the complete set of arguments as specified by the recipe.
    """

    @pytest.fixture
    def parser(self):
        parser = TuneArgumentParser("Test parser")
        return parser

    def test_configs(self, parser) -> None:
        for config_path, params in config_to_params.items():
            args, _ = parser.parse_known_args(["--config", config_path])
            try:
                _ = params(**vars(args))
            except ValueError as e:
                raise AssertionError(
                    f"Config {config_path} using params {params.__name__} is not well formed"
                ) from e
