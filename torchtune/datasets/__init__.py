# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset

from .alpaca import AlpacaDataset

DATASET_DICT = {"alpaca": AlpacaDataset}


def get_dataset(name: str, **kwargs) -> Dataset:
    if name in DATASET_DICT:
        return DATASET_DICT[name](**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def list_datasets():
    return list(DATASET_DICT)
