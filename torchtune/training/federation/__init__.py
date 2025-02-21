# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

from torchtune.training.federation._federator import DiLiCoFederator

from torchtune.training.federation._participant import TuneParticipant

Federator = Union[
    DiLiCoFederator,
]

Particpants = Union[
    TuneParticipant,
]

__all__ = ["TestFederator", "TuneParticipant"]
