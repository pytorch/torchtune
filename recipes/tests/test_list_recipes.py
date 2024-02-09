# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest
from recipes import list_recipes


class TestListRecipes:
    @pytest.fixture
    def dummy_recipes(self):
        return [
            "a.py",
            "b.py",
            "c.py",
            "__init__.py",
            "interfaces.py",
            "params.py",
        ]

    @mock.patch("recipes.os.listdir")
    def test_list_recipes(self, mock_listdir, dummy_recipes):
        mock_listdir.return_value = dummy_recipes
        assert list_recipes() == ["a", "b", "c"]
        mock_listdir.assert_called_once()
