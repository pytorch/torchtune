# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import functools
import sys
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf
from torchtune.utils.argparse import TuneArgumentParser
from torchtune.utils.logging import get_logger


Recipe = Callable[[DictConfig], Any]


def parse(function_to_run: Recipe) -> Callable[[Recipe], Any]:
    """
    Decorator that handles parsing the config file and CLI overrides
    for a recipe. Use it on the recipe's main function.
    """

    @functools.wraps(function_to_run)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        parser = TuneArgumentParser(
            description=function_to_run.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        # Get user-specified args from config and CLI and create params for recipe
        params, _ = parser.parse_known_args()
        params = OmegaConf.create(vars(params))

        logger = get_logger("DEBUG")
        logger.info(msg=f"Running {function_to_run.__name__} with parameters {params}")

        sys.exit(function_to_run(params))

    return wrapper
