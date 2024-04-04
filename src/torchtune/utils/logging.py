# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional


def get_logger(level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with a stream handler.

    Args:
        level (Optional[str]): The logging level. See https://docs.python.org/3/library/logging.html#levels for list of levels.

    Returns:
        logging.Logger: The logger.
    """
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())
    if level is not None:
        level = getattr(logging, level.upper())
        logger.setLevel(level)
    return logger
