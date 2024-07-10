# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Mapping, Protocol


class Transform(Protocol):
    """
    Loose interface for all data and model transforms. Transforms operate at the
    sample level and perform operations on a sample dict, returning the updated dict.
    """

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        pass
