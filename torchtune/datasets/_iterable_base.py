# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Iterator

from torch.utils.data import IterableDataset


class TuneIterableDataset(IterableDataset, ABC):
    """
    Abstract base class for all torchtune iterable datasets.
    It defines the minimal, consistent interface required for all dataset
    implementations to ensure they are compatible with the training loop,
    checkpointing, and metric logging systems.
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """A unique identifier for the dataset, used for namespacing in metrics and checkpoints."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Returns an infinite iterator over the dataset. Each implementation is responsible
        for its own iteration logic, including shuffling and making it an infinite stream.
        """
        pass

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Returns a state dictionary for checkpointing"""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from a state dictionary, used when resuming from a checkpoint."""
        pass
