# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import textwrap

from omegaconf import DictConfig
from torchtune import config
from torchtune.config._utils import _merge_yaml_and_cli_args
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
            $ tune validate --config recipes/configs/full_finetune_distributed.yaml
            Config is well-formed!
        """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Get user-specified args from config and CLI and create params for recipe
    yaml_args, cli_args = parser.parse_known_args()
    conf = _merge_yaml_and_cli_args(yaml_args, cli_args)

    main(conf)
