# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import textwrap
from pathlib import Path

from omegaconf import OmegaConf

from torchtune import config
from torchtune._cli.subcommand import Subcommand
from torchtune.config._errors import ConfigError


class Validate(Subcommand):
    """Holds all the logic for the `tune validate` subcommand."""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self._parser = subparsers.add_parser(
            "validate",
            prog="tune validate",
            help="Validate a config and ensure that it is well-formed.",
            description="Validate a config and ensure that it is well-formed.",
            usage="tune validate <config>",
            epilog=textwrap.dedent(
                """\
                examples:

                    $ tune validate recipes/configs/full_finetune_distributed.yaml
                    Config is well-formed!
                """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self._parser.set_defaults(func=self._validate_cmd)

    def _add_arguments(self) -> None:
        """Add arguments to the parser."""
        self._parser.add_argument(
            "config",
            type=Path,
            help="Path to a config to validate.",
        )

    def _validate_cmd(self, args: argparse.Namespace):
        """Validate a config file."""
        cfg = OmegaConf.load(args.config)

        try:
            config.validate(cfg)
        except ConfigError as e:
            self._parser.error(str(e))

        print("Config is well-formed!")
