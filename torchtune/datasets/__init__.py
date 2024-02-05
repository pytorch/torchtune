# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset

from .alpaca import AlpacaDataset
from .slimorca import SlimOrcaDataset

ALL_DATASETS = {"alpaca": AlpacaDataset, "slimorca": SlimOrcaDataset}


def get_dataset(name: str, **kwargs) -> Dataset:
    """Get known supported datasets by name"""
    if name in ALL_DATASETS:
        return ALL_DATASETS[name](**kwargs)
    else:
        raise ValueError(
            f"Dataset not recognized. Expected one of {ALL_DATASETS}, received {name}"
        )


def list_datasets():
    """List of availabe datasets supported by `get_dataset`"""
    return list(ALL_DATASETS)
