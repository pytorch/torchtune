# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from torch.utils.data import Dataset
from torchtune.trainer import ReproducibleDataLoader
from torchtune.utils.env import seed


class RandomTransformDataset(Dataset):
    def __init__(self, length: int):
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return torch.initial_seed()


def print_batches(base_seed, rank, num_workers):
    seed(base_seed + rank)
    dataset = RandomTransformDataset(4)
    dl = ReproducibleDataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: x,
        seed=base_seed,
    )
    for batch in dl:
        print(batch[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    print_batches(args.seed, args.rank, args.num_workers)
