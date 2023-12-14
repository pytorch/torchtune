# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset

from .alpaca import AlpacaDataset


def get_dataset(name: str, **kwargs) -> Dataset:
    if name == "alpaca":
        return AlpacaDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
