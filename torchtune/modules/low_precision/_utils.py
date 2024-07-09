# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib.metadata import PackageNotFoundError, version

import torchao


def _is_fbcode():
    return not hasattr(torch.version, "git_version")


def _get_torchao_version() -> Tuple[Optional[str], Optional[str]]:
    """
    Get torchao version. Returns a tuple of two elements, the first element
    is the version string, the second element is whether it's a nightly version.
    For fbcode usage, return None, None.

    Checks:
        1) is_fbcode, then
        2) torchao.__version__ (only defined for torchao >= 0.3.0), then
        3) importlib's version(torchao) for non-nightly, then
        4) importlib's version(torchao-nightly) for nightlies

    If none of these work, raise an error.

    """
    if _is_fbcode():
        return None, None
    try:
        ao_version = torchao.__version__
        is_nightly = False
    except AttributeError:
        ao_version = version("torchao")
        is_nightly = False
    # For importlib metadata, need to check nightly separately
    except PackageNotFoundError:
        ao_version = version("torchao-nightly")
        is_nightly = True
    except Exception as e:
        raise PackageNotFoundError("Could not find torchao version") from e
    return ao_version, is_nightly
