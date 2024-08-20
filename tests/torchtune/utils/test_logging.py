# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchtune.data._utils import deprecated


def test_deprecated():
    @deprecated(msg="Please use `TotallyAwesomeClass` instead.")
    class DummyClass:
        pass

    with pytest.warns(
        FutureWarning,
        match="DummyClass is deprecated and will be removed in future versions. Please use `TotallyAwesomeClass` instead.",
    ):
        DummyClass()

    with pytest.warns(None) as record:
        DummyClass()

    assert len(record) == 0, "Warning raised twice when it should only be raised once."

    @deprecated(msg="Please use `totally_awesome_func` instead.")
    def dummy_func():
        pass

    with pytest.warns(
        FutureWarning,
        match="dummy_func is deprecated and will be removed in future versions. Please use `totally_awesome_func` instead.",
    ):
        dummy_func()
