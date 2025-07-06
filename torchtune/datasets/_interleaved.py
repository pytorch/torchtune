# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
import logging
import math
from typing import Any, Iterator

import torch

from torchtune.datasets._iterable_base import (
    DatasetInfo,
    InfiniteTuneIterableDataset,
)

logger = logging.getLogger(__name__)


class InterleavedDataset(InfiniteTuneIterableDataset):
    """Infinitely interleaves multiple TuneIterableDatasets according to their sampling weights.
    - The weights are extracted from each dataset's info.weight property and normalized to sum to 1.0.
    - This dataset is responsible for managing the state of its child datasets
    to ensure correct checkpointing and resumption.

    Args:
        datasets (list[InfiniteTuneIterableDataset]): list of datasets to interleave.
        seed (int): Seed for sampling.
        weight (float): Weight for this dataset. Defaults to 1.0.
        dataset_name (str): Name of the dataset. Defaults to "interleaved_dataset".
        sampling_log_maxlen (int): Maximum length of the sampling log.
        
    Raises:
        ValueError: If duplicate dataset names are detected in the hierarchy.
    """

    def __init__(
        self,
        datasets: list[InfiniteTuneIterableDataset],
        seed: int,
        weight: float = 1.0,
        dataset_name: str = "interleaved_dataset",
        sampling_log_maxlen: int = 10000,
    ):
        self._datasets = sorted(datasets, key=lambda ds: ds.info.name)
        self._sampling_log_maxlen = sampling_log_maxlen

        # Build the hierarchical info object for this dataset
        self._info = DatasetInfo(
            name=dataset_name,
            weight=weight,
            children=tuple(ds.info for ds in self._datasets),
        )

        # Validate the entire hierarchy using the base class method
        self._validate_unique_dataset_names()

        # Extract weights from direct children and normalize them
        child_weights = [info.weight for info in self._info.children]
        total_weight = sum(child_weights)
        if not math.isclose(total_weight, 1.0, rel_tol=1e-9):
            logger.warning(
                f"Interleaved dataset normalized weights to sum to 1.0. "
                f"Previous weights={child_weights}, "
                f"new weights={[w / total_weight for w in child_weights]}"
            )
        self._normalized_weights = torch.tensor(
            [w / total_weight for w in child_weights], dtype=torch.float
        )
        
        # Track sampling decisions for debugging and analysis
        self._sampling_log: deque[tuple[int, str]] = deque(
            maxlen=self._sampling_log_maxlen
        )
        self._iteration_count = 0
        self._sampling_generator = torch.Generator().manual_seed(seed)

    @property
    def info(self) -> DatasetInfo:
        return self._info

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Interleave samples from child infinite datasets"""
        # Create a dictionary of iterators for each child dataset
        child_iters = {ds.info.name: iter(ds) for ds in self._datasets}

        while True:
            # Sample a child dataset based on the normalized weights
            ds_idx = torch.multinomial(
                self._normalized_weights,
                1,
                replacement=True,
                generator=self._sampling_generator,
            ).item()

            selected_ds = self._datasets[ds_idx]
            ds_name = selected_ds.info.name

            # Log
            self._sampling_log.append((self._iteration_count, ds_name))
            self._iteration_count += 1

            # Yield the next sample from the selected child iterator
            yield next(child_iters[ds_name])

    def state_dict(self) -> dict[str, Any]:
        """Save state for the interleaver and its children."""
        # The parent is responsible for namespacing the child states
        child_states = {ds.info.name: ds.state_dict() for ds in self._datasets}
        return {
            "sampling_generator_state": self._sampling_generator.get_state(),
            "child_states": child_states,
            "sampling_log": list(self._sampling_log),
            "iteration_count": self._iteration_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state for the interleaver and its children."""
        self._sampling_generator.set_state(state_dict["sampling_generator_state"])
        child_states = state_dict["child_states"]

        for ds in self._datasets:
            ds.load_state_dict(child_states[ds.info.name])

        # Load sampling log and iteration count
        self._sampling_log = deque(
            state_dict.get("sampling_log", []), maxlen=self._sampling_log_maxlen
        )
        self._iteration_count = state_dict.get("iteration_count", 0)
