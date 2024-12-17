# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Callable, Iterable, Iterator, Mapping, TypeVar

from torchtune.utils._import_guard import _TORCHDATA_INSTALLED, _TORCHDATA_MIN_VERSION

from typing_extensions import TypeAlias


if _TORCHDATA_INSTALLED:
    from torchdata.nodes import BaseNode, Loader  # noqa
else:
    # If we fail to import torchdata, define stubs to make typechecker happy
    T = TypeVar("T")

    class BaseNode(Iterator[T]):
        def __init__(self, *args, **kwargs):
            pass

    class Loader(Iterable):
        def __init__(self, *args, **kwargs):
            assert_torchdata_installed()


DatasetType: TypeAlias = BaseNode[Mapping[str, Any]]  # type: ignore


def assert_torchdata_installed():
    if not _TORCHDATA_INSTALLED:
        raise ImportError(
            f"torchdata is not installed, or the current version is too old. "
            f"Please (re-)install it with `pip install torchdata>={_TORCHDATA_MIN_VERSION}`. "
        )


def requires_torchdata(func: Callable) -> Callable:
    """
    Decorator to check if torchdata is installed and raise an ImportError if not.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        assert_torchdata_installed()
        return func(*args, **kwargs)

    return wrapper
