# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping, Protocol


class Transform(Protocol):
    """
    Loose interface for all data and model transforms. Transforms operate at the
    sample level and perform operations on a sample dict which is contained in
    kwargs. Any fields that will be processed are unfolded with explicit keyword-arguments,
    then the updated dict is returned.
    """

    def __call__(self, **kwargs) -> Mapping[str, Any]:
        pass


class Compose(Transform):
    """
    Compose multiple transforms together, inspired by torchvision's ``Compose`` API
    """

    def __init__(self, transforms: List[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, **kwargs) -> Mapping[str, Any]:
        for transform in self.transforms:
            kwargs = transform(**kwargs)
        return kwargs
