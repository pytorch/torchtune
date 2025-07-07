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
    """Hierarchical metadata for datasets, enabling composition and weight tracking.

    Used to build tree structures when composing datasets. For example, a nested
    `InterleavedDataset` dataset would have this structure:

    Example:
    .. code-block:: python

        DatasetInfo(name='parent_interleaved',
            weight=1.0,
            children=(DatasetInfo(name='child_interleaved',
                                  weight=0.7,
                                  children=(DatasetInfo(name='dataset_a',
                                                        weight=0.6,
                                                        children=()),
                                            DatasetInfo(name='dataset_b',
                                                        weight=0.4,
                                                        children=()))),
                      DatasetInfo(name='dataset_c', weight=0.3, children=())))

    This hierarchical structure is used for validation (ensuring unique dataset
    names) and for logging metrics.

    Attributes:
        name (str): Unique identifier for the dataset
        weight (float): Sampling weight for dataset selection (default: 1.0)
        children (tuple[DatasetInfo, ...]): Nested datasets for composed structures
    """

    name: str
    weight: float = 1.0
    children: tuple["DatasetInfo", ...] = field(default_factory=tuple)


class TuneIterableDataset(IterableDataset, ABC):
    """Base class for all torchtune iterable datasets.

    Datasets are composable, enabling complex structures such as:
    ``PackedDataset(InterleavedDataset([InterleavedDataset([ds1, ds2]), ds3]))``

    Each dataset implementation must:
    - Track hierarchical metadata via the ``info`` property
    - Ensure unique dataset names across the entire tree
    - Handle checkpointing: parents resume children's state
    - Provide proper state management for exact resumption
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
        """Returns checkpoint state for dataset resumption."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores dataset state from checkpoint."""
        pass


class InfiniteTuneIterableDataset(TuneIterableDataset):
    """Base class for infinite datasets that never exhaust.

    Prevents distributed training hangs by ensuring all ranks always
    have data available. Datasets restart from beginning when exhausted.
    """

    pass
