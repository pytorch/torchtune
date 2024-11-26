import functools
from typing import Any, Callable, Iterable, Iterator, Mapping

from typing_extensions import TypeAlias  # typing.TypeAlias is only in Python 3.10+


try:
    from torchdata.nodes import BaseNode, Loader  # noqa

    _TORCHDATA_INSTALLED = True
    DatasetType: TypeAlias = BaseNode[Mapping[str, Any]]  # type: ignore
except ImportError as e:
    # If we fail to import torchdata, define some stubs to make typechecker happy
    _TORCHDATA_INSTALLED = False
    DatasetType: TypeAlias = Iterator[Mapping[str, Any]]  # type: ignore

    class Loader(Iterable):
        def __init__(self, *args, **kwargs):
            assert_torchdata_installed()


MIN_VERSION = "0.10.0"


def assert_torchdata_installed():
    if not _TORCHDATA_INSTALLED:
        raise ImportError(
            f"torchdata is not installed, or the current version is too old. "
            f"Please (re-)install it with `pip install torchdata>={MIN_VERSION}`. "
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
