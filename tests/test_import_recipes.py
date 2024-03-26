# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest


def test_import_receipes():
    with pytest.raises(ModuleNotFoundError, match="No module named 'recipes'"):
        import recipes  # noqa
