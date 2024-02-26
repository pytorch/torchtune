# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchtune.config._utils import get_object_from_path

class TestUtils:
    def test_get_object_from_path(self):
        good_paths = [
            "torchtune",  # Test single module without dot
            "torchtune.models",  # Test dotpath for a module
            "torchtune.models.llama2_7b",  # Test dotpath for an object
        ]
        for path in good_paths:
            _ = get_object_from_path(path)

        # Test that a relative path fails
        with pytest.raises(ValueError, match="Relative imports are not supported"):
            _ = get_object_from_path(".test")
        # Test that a non-existent path fails
        with pytest.raises(ImportError, match="Error loading 'torchtune.models.dummy'"):
            _ = get_object_from_path("torchtune.models.dummy")
