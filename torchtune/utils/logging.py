# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
from functools import lru_cache, wraps
from typing import Callable, Optional, TypeVar

T = TypeVar("T", bound=type)


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


def deprecated(msg: str = "") -> Callable[[T], T]:
    """
    Decorator to mark an object as deprecated and print additional message.

    Args:
        msg (str): additional information to print after warning.

    Returns:
        Callable[[T], T]: the decorated object.
    """

    @lru_cache(maxsize=1)
    def warn(obj):
        warnings.warn(
            f"{obj.__name__} is deprecated and will be removed in future versions. "
            + msg,
            category=FutureWarning,
            stacklevel=3,
        )

    def decorator(obj):
        @wraps(obj)
        def wrapper(*args, **kwargs):
            warn(obj)
            return obj(*args, **kwargs)

        return wrapper

    return decorator
