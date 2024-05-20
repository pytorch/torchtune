# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import deeplake
from torch.utils.data import Dataset
from torchtune import utils

log = utils.get_logger("DEBUG")
# """
# A PyTorch Dataset class for loading data from ActiveLoop's DeepLake platform.

# This class serves as a data loader for working with datasets stored in ActiveLoop's DeepLake platform.
# It takes a DeepLake dataset object as input and provides functionality to load data from it
# using PyTorch's DataLoader interface.

# Attributes:
#     ds (deeplake.Dataset): The dataset object obtained from ActiveLoop's DeepLake platform.

# Methods:
#     __init__(self, ds: deeplake.Dataset)
#         Initializes the DeepLakeDataloader with the given dataset object.

#     Args:
#         ds (deeplake.Dataset): The dataset object obtained from ActiveLoop's DeepLake platform.

# Example:
#     # Load a dataset from DeepLake and create a DataLoader
#     dataset = load_deeplake_dataset("my_deep_lake_dataset")
#     dataloader = DeepLakeDataloader(dataset)
# """


class DeepLakeDataloader(Dataset):
    """A PyTorch Dataset class for loading data from ActiveLoop's DeepLake platform.

    This class serves as a data loader for working with datasets stored in ActiveLoop's DeepLake platform.
    It takes a DeepLake dataset object as input and provides functionality to load data from it
    using PyTorch's DataLoader interface.

    Args:
        ds (deeplake.Dataset): The dataset object obtained from ActiveLoop's DeepLake platform.
    """

    def __init__(self, ds: deeplake.Dataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        column_map = self.ds.tensors.keys()

        values_dataset = {}
        for el in column_map:  # {"column_name" : value}
            values_dataset[el] = self.ds[el][idx].text().astype(str)

        return values_dataset


def load_deep_lake_dataset(
    deep_lake_dataset: str, **config_kwargs
) -> DeepLakeDataloader:
    """
    Load a dataset from ActiveLoop's DeepLake platform.

    Args:
        deep_lake_dataset (str): The name of the dataset to load from DeepLake.
        **config_kwargs: Additional keyword arguments passed to `deeplake.dataset`.

    Returns:
        DeepLakeDataloader: A data loader for the loaded dataset.
    """
    ds = deeplake.dataset(deep_lake_dataset, **config_kwargs)
    log.info(f"Dataset loaded from deeplake: {ds}")
    return DeepLakeDataloader(ds)
