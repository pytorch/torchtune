# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchtune.utils.config import validate_recipe_args


def _fake_recipe(a: str, b: int, c: float, d: bool):
    pass


class TestConfig:
    def test_validate_recipe_args(self):
        args = {"a": "1", "b": 2, "c": 3.0, "d": True}
        validate_recipe_args(_fake_recipe, args)

        with pytest.raises(ValueError):
            args = {"a": "1", "b": 2, "d": True}
            validate_recipe_args(_fake_recipe, args)

        with pytest.raises(ValueError):
            args = {"a": "1", "b": 2, "c": 3.0, "d": True, "e": 4}
            validate_recipe_args(_fake_recipe, args)

        with pytest.raises(TypeError):
            args = {"a": 1, "b": 2, "c": 3.0, "d": True}
            validate_recipe_args(_fake_recipe, args)
