# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from io import StringIO
from unittest import mock

import pytest
from torchtune.utils._logging import deprecate_parameter, deprecated, log_rank_zero


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


def test_deprecate_parameter():
    @deprecate_parameter(param_name="param_a", msg="Please use param_b instead.")
    class DummyClass:
        def __init__(self, param_a, param_b=None):
            pass

    with pytest.warns(
        FutureWarning,
        match="param_a is deprecated for DummyClass and will be removed in future versions. Please use param_b instead.",
    ):
        DummyClass(1)

    with pytest.warns(None) as record:
        DummyClass(1)

    assert len(record) == 0, "Warning raised twice when it should only be raised once."

    @deprecate_parameter(param_name="param_a", msg="Please use param_b instead.")
    def dummy_func(param_a, param_b=None):
        pass

    with pytest.warns(
        FutureWarning,
        match="param_a is deprecated for dummy_func and will be removed in future versions. Please use param_b instead.",
    ):
        dummy_func(1)


def test_log_rank_zero(capsys):
    # Create a logger and add a StreamHandler to it so we can
    # assert on logged strings
    logger = logging.getLogger(__name__)
    logger.setLevel("DEBUG")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)

    with (
        mock.patch("torchtune.utils._logging.dist.is_available", return_value=True),
        mock.patch("torchtune.utils._logging.dist.is_initialized", return_value=True),
    ):
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


def test_log_rank_zero_before_dist_init(capsys):
    """log_rank_zero uses RANK env var when distributed is not yet initialized.

    torchrun sets RANK before recipe_main runs, so config.log_config (which
    calls log_rank_zero) was printing on every rank because dist.is_initialized()
    was False and the old code fell back to rank=0 unconditionally.
    """
    logger = logging.getLogger(__name__ + ".pre_init")
    logger.setLevel("DEBUG")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)

    not_initialized = mock.patch(
        "torchtune.utils._logging.dist.is_initialized", return_value=False
    )
    dist_available = mock.patch(
        "torchtune.utils._logging.dist.is_available", return_value=True
    )

    with not_initialized, dist_available:
        # RANK=0 in env → rank-zero process should log
        with mock.patch.dict(os.environ, {"RANK": "0"}):
            stream.truncate(0)
            stream.seek(0)
            log_rank_zero(logger, "rank zero pre-init", level=logging.DEBUG)
            assert "rank zero pre-init" in stream.getvalue()

        # RANK=1 in env → non-zero process should stay silent
        with mock.patch.dict(os.environ, {"RANK": "1"}):
            stream.truncate(0)
            stream.seek(0)
            log_rank_zero(logger, "rank one pre-init", level=logging.DEBUG)
            assert not stream.getvalue().strip()

        # RANK not set (single-device training) → default rank=0, should log
        env_without_rank = {k: v for k, v in os.environ.items() if k != "RANK"}
        with mock.patch.dict(os.environ, env_without_rank, clear=True):
            stream.truncate(0)
            stream.seek(0)
            log_rank_zero(logger, "single device", level=logging.DEBUG)
            assert "single device" in stream.getvalue()
