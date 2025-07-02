# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
import math
from collections import deque
from typing import Any, Iterator

import torch

from torchtune.datasets._iterable_base import TuneIterableDataset

logger = logging.getLogger(__name__)


class InterleavedDataset(TuneIterableDataset):
    """Infinitely interleaves multiple TuneIterableDatasets according to their sampling weights.
    - The weights are extracted from each dataset's sampling_weight property and normalized to sum to 1.0.
    - This dataset is responsible for managing the state of its child datasets
    to ensure correct checkpointing and resumption.

    Args:
        datasets (list[TuneIterableDataset]): list of TuneIterableDatasets to interleave.
        seed (int): Seed for sampling.
        dataset_name (str): Name of the dataset. If None, defaults to "interleaved_dataset".
        sampling_log_maxlen (int): Maximum length of the sampling log.
        
    Raises:
        ValueError: If duplicate dataset names are detected in the provided datasets.
    """

    def __init__(
        self,
        datasets: list[TuneIterableDataset],
        seed: int,
        dataset_name: str = "interleaved_dataset",
        sampling_log_maxlen: int = 10000,
    ):
        self._dataset_name = dataset_name
        self._sampling_log_maxlen = sampling_log_maxlen

        # Preserve original order for weighted sampling
        self._dataset_names = [ds.dataset_name for ds in datasets]

        # Create a name-to-dataset mapping for robust state management
        self._datasets: dict[str, TuneIterableDataset] = {
            ds.dataset_name: ds for ds in datasets
        }

        # Validate unique dataset names upfront - fail fast with clear error
        names = self._dataset_names
        if len(names) != len(set(names)):
            duplicates = [
                name for name, count in collections.Counter(names).items() if count > 1
            ]
            raise ValueError(
                f"Duplicate dataset names detected: {duplicates}. All {names=}"
                f"Please provide a unique 'dataset_name' for each dataset in the interleaved list."
            )

        self._sampling_generator = torch.Generator().manual_seed(seed)

        # Track sampling decisions for debugging and analysis
        self._sampling_log: deque[tuple[int, str]] = deque(maxlen=self._sampling_log_maxlen)
        self._iteration_count = 0

        # Extract weights from datasets' sampling_weight property
        weights = []
        for ds in datasets:
            weight = ds.sampling_weight
            if isinstance(weight, dict):
                # For composite datasets, sum up their weights
                weight = sum(weight.values())
            weights.append(weight)

        # Normalize weights to sum to 1
        total_weight = sum(weights)
        self._weights = torch.tensor(
            [w / total_weight for w in weights], dtype=torch.float
        )
        if not math.isclose(total_weight, 1.0, rel_tol=1e-9):
            logger.warning(
                f"Interleaved dataset normalized weights to sum to 1.0. "
                f"Found {total_weight=}. Previous {weights=}, new {self._weights.tolist()}"
            )

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def sampling_weight(self) -> dict[str, float]:
        return {name: weight.item() for name, weight in zip(self._dataset_names, self._weights)}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Interleave samples from child infinite datasets"""
        child_iters = {name: iter(ds) for name, ds in self._datasets.items()}

        while True:
            # Sample which dataset to use
            ds_idx = torch.multinomial(
                self._weights, 1, replacement=True, generator=self._sampling_generator
            ).item()

            # Sample an index, then get the name for safe lookup
            ds_name = self._dataset_names[ds_idx]

            # Log this sampling decision
            self._sampling_log.append((self._iteration_count, ds_name))
            self._iteration_count += 1

            try:
                sample = next(child_iters[ds_name])
                yield sample
            except StopIteration:
                # Per the design, child datasets must be infinite.
                # We re-initialize to allow for continuous operation but warn loudly
                # as this may indicate a design problem in the child dataset.
                logger.warning(
                    f"Child dataset {self._datasets[ds_name].dataset_name} was exhausted. "
                    "This is unexpected for an infinite dataset. Re-initializing its iterator."
                )
                child_iters[ds_name] = iter(self._datasets[ds_name])
                sample = next(child_iters[ds_name])
                yield sample

    def state_dict(self) -> dict[str, Any]:
        """Save state for the interleaver and its children."""
        # The parent is responsible for namespacing the child states.
        child_states = {name: ds.state_dict() for name, ds in self._datasets.items()}
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
        for name, ds in self._datasets.items():
            if name in child_states:
                # Pass the raw state dict to the child
                ds.load_state_dict(child_states[name])
        
        # Load sampling log and iteration count
        self._sampling_log = deque(
            state_dict.get("sampling_log", []), maxlen=self._sampling_log_maxlen
        )
        self._iteration_count = state_dict.get("iteration_count", 0)
