# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

_SCRIPTS = ["download", "convert_checkpoint", "ls", "cp", "eval", "validate"]


def list_scripts():
    """List of available scripts."""
    return _SCRIPTS
