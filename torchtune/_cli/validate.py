# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import textwrap

from omegaconf import DictConfig, OmegaConf
from torchtune import config
from torchtune.utils import TuneArgumentParser


def main(cfg: DictConfig):
    config.validate(cfg)
    print("Config is well-formed!")


if __name__ == "__main__":
    parser = TuneArgumentParser(
        description="Validate a config and ensure that it is well-formed.",
        usage="tune validate",
        epilog=textwrap.dedent(
            """\
        examples:
            $ tune validate --config recipes/configs/alpaca_llama2_full_finetune.yaml
            Config is well-formed!
        """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Get user-specified args from config and CLI and create params for recipe
    params, _ = parser.parse_known_args()
    params = OmegaConf.create(vars(params))

    main(params)
