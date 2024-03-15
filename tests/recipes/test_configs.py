# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from pathlib import Path

from omegaconf import OmegaConf
from torchtune import config

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "recipes" / "configs"


class TestConfigs:
    def test_instantiate(self) -> None:
        all_configs = [
            os.path.join(CONFIG_DIR, f)
            for f in os.listdir(CONFIG_DIR)
            if f.endswith(".yaml")
        ]
        for config_path in all_configs:
            cfg = OmegaConf.load(config_path)
            config.validate(cfg)
