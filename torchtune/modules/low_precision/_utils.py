# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Optional, Tuple

import torch

import torchao


def _is_fbcode():
    return not hasattr(torch.version, "git_version")


def _nightly_version_ge(ao_version_str: str, date: str) -> bool:
    """
    Compare a torchao nightly version to a date of the form
    %Y-%m-%d.

    Returns True if the nightly version is greater than or equal to
        the date, False otherwise
    """
    ao_datetime = datetime.strptime(ao_version_str.split("+")[0], "%Y.%m.%d")
    return ao_datetime >= datetime.strptime(date, "%Y-%m-%d")


def _get_torchao_version() -> Tuple[Optional[str], Optional[bool]]:
    """
    Get torchao version. Returns a tuple of two elements, the first element
    is the version string, the second element is whether it's a nightly version.
    For fbcode usage, return None, None.

    Checks:
        1) is_fbcode, then
        2) importlib's version(torchao-nightly) for nightlies, then
        3) torchao.__version__ (only defined for torchao >= 0.3.0), then
        4) importlib's version(torchao) for non-nightly


    If none of these work, raise an error.

    """
    if _is_fbcode():
        return None, None
    # Check for nightly install first
    try:
        ao_version = version("torchao-nightly")
        is_nightly = True
    except PackageNotFoundError:
        try:
            ao_version = torchao.__version__
            is_nightly = False
        except AttributeError:
            ao_version = "unknown"
    if ao_version == "unknown":
        try:
            ao_version = version("torchao")
            is_nightly = False
        except Exception as e:
            raise PackageNotFoundError("Could not find torchao version") from e
    return ao_version, is_nightly
