# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime

import torch


def torch_version_ge(version: str) -> bool:
    """
    Check if torch version is greater than or equal to the given version.

    Args:
        version (str): The torch version to compare against

    Returns:
        bool: True if torch version is greater than or equal to the given version.

    Example:
        >>> print(torch.__version__)
        2.4.0
        >>> torch_version_ge("2.0")
        True
    """
    return version in torch.__version__ or torch.__version__ >= version


def _is_fbcode():
    return not hasattr(torch.version, "git_version")


def _nightly_version_ge(ao_version_str: str, date: str) -> bool:
    """
    Compare a torchao nightly version to a date of the form
    %Y-%m-%d.

    Returns True if the nightly version is greater than or equal to
        the date, False otherwise
    """
    ao_datetime = datetime.strptime(
        ao_version_str.split("+")[0].split("dev")[1], "%Y%m%d"
    )
    return ao_datetime >= datetime.strptime(date, "%Y-%m-%d")
