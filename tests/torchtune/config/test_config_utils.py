# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from io import StringIO
from unittest import mock

import pytest
from omegaconf import OmegaConf
from torchtune.config._parse import TuneRecipeArgumentParser
from torchtune.config._utils import (
    _get_component_from_path,
    _merge_yaml_and_cli_args,
    _remove_key_by_dotpath,
    InstantiationError,
    log_config,
)

_CONFIG = {
    "a": 1,
    "b": {
        "_component_": 2,
        "c": 3,
    },
    "d": 4,
    "f": 8,
    "g": "foo",
    "h": "${g}/bar",
}


# Test local function
def local_fn():
    return "hello world"


class TestUtils:
    def test_get_component_from_path(self):
        # Test valid paths with three components
        valid_paths = [
            "torchtune",
            "os.path.join",
        ]
        for path in valid_paths:
            result = _get_component_from_path(path)
            assert result is not None, f"Failed to resolve valid path '{path}'"

        # test callable
        path = "torchtune.models.llama2.llama2_7b"
        result = _get_component_from_path(path)
        assert callable(result), f"Resolved '{path}' is not callable"

        # simulate call from globals
        fn = _get_component_from_path("local_fn")
        output = fn()
        assert output == "hello world", f"Got {output=}. Expected 'hello world'."

        # Test empty path
        with pytest.raises(InstantiationError, match="Invalid path: ''"):
            _get_component_from_path("")

        # Test non-string path
        with pytest.raises(InstantiationError, match="Invalid path: '123'"):
            _get_component_from_path(123)

        # Test relative imports
        relative_paths = [
            ".test.module",  # Leading dot
            "test.module.",  # Trailing dot
            "test..module",  # Consecutive dots
        ]
        for path in relative_paths:
            with pytest.raises(
                ValueError,
                match="Invalid dotstring. Relative imports are not supported.",
            ):
                _get_component_from_path(path)

        # Test non-existent components
        # Single-part path not found
        with pytest.raises(
            InstantiationError,
            match=r"Could not resolve 'nonexistent': not a module and not found in the caller's globals\.",
        ):
            _get_component_from_path("nonexistent")

        # Multi-part path with import failure
        with pytest.raises(
            InstantiationError, match=r"Could not import module 'os\.nonexistent': .*"
        ):
            _get_component_from_path("os.nonexistent.attr")

        # Multi-part path with attribute error
        with pytest.raises(
            InstantiationError,
            match=r"Module 'os\.path' has no attribute 'nonexistent'",
        ):
            _get_component_from_path("os.path.nonexistent")

    @mock.patch(
        "torchtune.config._parse.OmegaConf.load", return_value=OmegaConf.create(_CONFIG)
    )
    def test_merge_yaml_and_cli_args(self, mock_load):
        parser = TuneRecipeArgumentParser("test parser")
        yaml_args, cli_args = parser.parse_known_args(
            [
                "--config",
                "test.yaml",
                "b.c=4",  # Test overriding a flat param in a component
                "b=5",  # Test overriding component path
                "b.b.c=6",  # Test nested dotpath
                "d=6",  # Test overriding a flat param
                "e=7",  # Test adding a new param
                "~f",  # Test removing a param
                "g=bazz",  # Test interpolation happens after override
            ]
        )
        conf = _merge_yaml_and_cli_args(yaml_args, cli_args)
        assert conf.a == 1, f"a == {conf.a}, not 1 as set in the config."
        assert (
            conf.b._component_ == 5
        ), f"b == {conf.b._component_}, not 5 as set in overrides."
        assert conf.b.c == 4, f"b.c == {conf.b.c}, not 4 as set in overrides."
        assert conf.b.b.c == 6, f"b.b.c == {conf.b.b.c}, not 6 as set in overrides."
        assert conf.d == 6, f"d == {conf.d}, not 6 as set in overrides."
        assert conf.e == 7, f"e == {conf.e}, not 7 as set in overrides."
        assert "f" not in conf, f"f == {conf.f}, not removed as set in overrides."
        assert conf.h == "bazz/bar", f"h == {conf.h}, not bazz/bar as set in overrides."
        mock_load.assert_called_once()

        yaml_args, cli_args = parser.parse_known_args(
            [
                "--config",
                "test.yaml",
                "b=5",  # Test overriding component path but keeping other kwargs
            ]
        )
        conf = _merge_yaml_and_cli_args(yaml_args, cli_args)
        assert (
            conf.b._component_ == 5
        ), f"b == {conf.b._component_}, not 5 as set in overrides."
        assert conf.b.c == 3, f"b.c == {conf.b.c}, not 3 as set in the config."
        assert mock_load.call_count == 2

        yaml_args, cli_args = parser.parse_known_args(
            [
                "--config",
                "test.yaml",
                "b.c=5",  # Test overriding kwarg but keeping component path
            ]
        )
        conf = _merge_yaml_and_cli_args(yaml_args, cli_args)
        assert (
            conf.b._component_ == 2
        ), f"b == {conf.b._component_}, not 2 as set in the config."
        assert conf.b.c == 5, f"b.c == {conf.b.c}, not 5 as set in overrides."
        assert mock_load.call_count == 3

        yaml_args, cli_args = parser.parse_known_args(
            [
                "--config",
                "test.yaml",
                "b",  # Test invalid override
            ]
        )
        with pytest.raises(
            ValueError, match="Command-line overrides must be in the form of key=value"
        ):
            _ = _merge_yaml_and_cli_args(yaml_args, cli_args)

    def test_log_config(self, capsys):
        cfg = OmegaConf.create({"test": {"a": 1, "b": 2}})

        # Create a logger and add a StreamHandler to it so we can patch the
        # config logger and assert on logged strings
        logger = logging.getLogger(__name__)
        logger.setLevel("DEBUG")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)

        with mock.patch(
            "torchtune.config._utils.get_logger", return_value=logger
        ), mock.patch(
            "torchtune.utils._logging.dist.is_available", return_value=True
        ), mock.patch(
            "torchtune.utils._logging.dist.is_initialized", return_value=True
        ):
            # Make sure rank 0 logs as expected
            with mock.patch(
                "torchtune.utils._logging.dist.get_rank",
                return_value=0,
            ):
                log_config("test", cfg)
                output = stream.getvalue().strip()
                assert (
                    "Running test with resolved config:\n\ntest:\n  a: 1\n  b: 2"
                    in output
                )

            # Clear the stream
            stream.truncate(0)
            stream.seek(0)

            # Make sure all other ranks do not log anything
            with mock.patch(
                "torchtune.utils._logging.dist.get_rank",
                return_value=1,
            ):
                log_config("test", cfg)
                output = stream.getvalue().strip()
                assert not output

    def test_remove_key_by_dotpath(self):
        # Test removing a component raises
        cfg = copy.deepcopy(_CONFIG)
        with pytest.raises(
            ValueError, match="Removing components from CLI is not supported"
        ):
            _remove_key_by_dotpath(cfg, "b")

        # Test removing a top-level param
        cfg = copy.deepcopy(_CONFIG)
        _remove_key_by_dotpath(cfg, "a")
        assert "a" not in cfg

        # Test removing a component param
        cfg = copy.deepcopy(_CONFIG)
        _remove_key_by_dotpath(cfg, "b.c")
        assert "c" not in cfg["b"]

        # Test removing nested one level too deep fails
        cfg = copy.deepcopy(_CONFIG)
        with pytest.raises(TypeError, match="'int' object is not subscriptable"):
            _remove_key_by_dotpath(cfg, "b.c.d")

        # Test removing non-existent param fails
        cfg = copy.deepcopy(_CONFIG)
        with pytest.raises(KeyError, match="'i'"):
            _remove_key_by_dotpath(cfg, "i")
