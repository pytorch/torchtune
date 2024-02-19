# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Protocol


class Params(Protocol):
    """
    Interface for recipe params dataclasses. Exposes validate method that is
    automatically called in post init.
    """

    def validate(self) -> None:
        """
        User input validation for all params attributes. Place any validation
        logic here.
        """
        ...

    def __post_init__(self) -> None:
        """
        Automatically called after init.
        """
        self.validate()
