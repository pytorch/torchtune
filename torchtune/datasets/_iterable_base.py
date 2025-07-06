# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

from torch.utils.data import IterableDataset


@dataclass(frozen=True)
class DatasetInfo:
    """Represents hierarchical information about a dataset, including its name,
    sampling weight and children. Children is a common case when composing datasets,
    e.g. Packed(InterleavedDataset([ds1, ds2])).
    """

    name: str
    weight: float = 1.0
    children: tuple["DatasetInfo", ...] = field(default_factory=tuple)


class TuneIterableDataset(IterableDataset, ABC):
    """Abstract base class for all torchtune iterable datasets.
    It defines the minimal, consistent interface required for all dataset
    implementations to ensure they are compatible with the training loop,
    checkpointing, and metric logging systems.
    """

    @property
    @abstractmethod
    def info(self) -> DatasetInfo:
        """Returns a hierarchical structure of all dataset information, including
        this dataset and its children."""
        pass

    def _validate_unique_dataset_names(self) -> None:
        """Traverses the DatasetInfo tree and raises ValueError on duplicate names."""
        root_info = self.info
        names = []
        to_process = [root_info]

        while to_process:
            node = to_process.pop(0)
            names.append(node.name)
            to_process.extend(node.children)

        # Check for duplicates after traversing the whole tree
        duplicates = [name for name in set(names) if names.count(name) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate dataset names found in hierarchy: {duplicates=}, all names={names}"
            )

    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Returns an iterator over the dataset. Each implementation is responsible
        for its own iteration logic, including shuffling, distribution of data across ranks,
        and making it an infinite stream."""
        pass

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Returns a state dictionary for checkpointing"""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from a state dictionary, used when resuming from a checkpoint."""
        pass


class InfiniteTuneIterableDataset(TuneIterableDataset):
    """Abstract base class for infinite datasets, which yield samples indefinitely.
    It only purpose is to make it explicit that the dataset is expected to be infinite, i.e.
    it never exhausts. This is helpful to avoid complexity due to some rank hanging because
    of lack of data"""

    pass
