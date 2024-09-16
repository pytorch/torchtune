# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from io import StringIO
from unittest import mock

import pytest
from torchtune.utils._logging import deprecated, log_rank_zero


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


def test_log_rank_zero(capsys):
    # Create a logger and add a StreamHandler to it so we can
    # assert on logged strings
    logger = logging.getLogger(__name__)
    logger.setLevel("DEBUG")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)

    with mock.patch(
        "torchtune.utils._logging.dist.is_available", return_value=True
    ), mock.patch("torchtune.utils._logging.dist.is_initialized", return_value=True):
        # Make sure rank 0 logs as expected
        with mock.patch(
            "torchtune.utils._logging.dist.get_rank",
            return_value=0,
        ):
            log_rank_zero(logger, "this is a test", level=logging.DEBUG)
            output = stream.getvalue().strip()
            assert "this is a test" in output

        # Clear the stream
        stream.truncate(0)
        stream.seek(0)

        # Make sure all other ranks do not log anything
        with mock.patch(
            "torchtune.utils._logging.dist.get_rank",
            return_value=1,
        ):
            log_rank_zero(logger, "this is a test", level=logging.DEBUG)
            output = stream.getvalue().strip()
            assert not output
