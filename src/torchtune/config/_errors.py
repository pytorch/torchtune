# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List


class InstantiationError(Exception):
    """
    Raised when a `_component_` field in a config is unable to be instantiated.
    """

    pass


class ConfigError(Exception):
    """
    Raised when the yaml config is not well-formed. Prints all the collected
    errors at once.

    Args:
        errors (List[Exception]): exceptions found when validating `_component_`
            fields in the config
    """

    def __init__(self, errors: List[Exception]):
        self.errors = errors

    def __str__(self):
        error_messages = [f"{type(e).__name__}: {str(e)}" for e in self.errors]
        return "Config is not well-formed, found the following errors: \n" + "\n".join(
            error_messages
        )
