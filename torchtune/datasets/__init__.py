# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset

from .alpaca import AlpacaDataset
from .slimorca import SlimOrcaDataset  # noqa
from .stateful_dataloader import StatefulDataLoader  # noqa

_DATASET_DICT = {"alpaca": AlpacaDataset, "slimorca": SlimOrcaDataset}


def get_dataset(name: str, **kwargs) -> Dataset:
    """Get known supported datasets by name"""
    if name in _DATASET_DICT:
        return _DATASET_DICT[name](**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def list_datasets():
    """List of availabe datasets supported by `get_dataset`"""
    return list(_DATASET_DICT)
