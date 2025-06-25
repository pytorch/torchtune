import collections
import logging
import math
from typing import Any, Dict, Iterator, List

import torch

from torchtune.datasets._iterable_base import TuneIterableDataset

logger = logging.getLogger(__name__)


class InterleavedDataset(TuneIterableDataset):
    """Infinitely interleaves multiple TuneIterableDatasets according to a list of weights.
    - The weights are normalized to sum to 1.0.
    - This dataset is responsible for managing the state of its child datasets
    to ensure correct checkpointing and resumption.

    Args:
        datasets (List[TuneIterableDataset]): List of TuneIterableDatasets to interleave.
        weights (List[float]): List of weights for each dataset. Must sum to 1.0.
        seed (int): Seed for sampling.
        dataset_name (str): Name of the dataset. If None, defaults to "interleaved_dataset".
    """

    def __init__(
        self,
        datasets: List[TuneIterableDataset],
        weights: List[float],
        seed: int,
        dataset_name: str = "interleaved_dataset",
    ):
        self._dataset_name = dataset_name

        # Preserve original order for weighted sampling
        self._dataset_names = [ds.dataset_name for ds in datasets]

        # Create a name-to-dataset mapping for robust state management
        self._datasets: Dict[str, TuneIterableDataset] = {
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

        # Normalize weights to sum to 1
        #TODO: make it a property? rely on ds.weight? 
        total_weight = sum(weights)
        self._weights = torch.tensor(
            [w / total_weight for w in weights], dtype=torch.float
        )
        if not math.isclose(total_weight, 1.0, rel_tol=1e-9):
            logger.warning(
                f"Interleaved dataset normalized weights to sum to 1.0. Found {total_weight=}. Previous {weights=}, new {self._weights.tolist()}"
            )

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Interleave samples from child infinite datasets"""
        child_iters = {name: iter(ds) for name, ds in self._datasets.items()}

        while True:
            # Sample which dataset to use
            ds_idx = torch.multinomial(
                self._weights, 1, replacement=True, generator=self._sampling_generator
            ).item()

            # Sample an index, then get the name for safe lookup
            ds_name = self._dataset_names[ds_idx]

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

    def state_dict(self) -> Dict[str, Any]:
        """Save state for the interleaver and its children."""
        # The parent is responsible for namespacing the child states.
        child_states = {name: ds.state_dict() for name, ds in self._datasets.items()}
        return {
            "sampling_generator_state": self._sampling_generator.get_state(),
            "child_states": child_states,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state for the interleaver and its children."""
        self._sampling_generator.set_state(state_dict["sampling_generator_state"])
        child_states = state_dict["child_states"]
        for name, ds in self._datasets.items():
            if name in child_states:
                # Pass the raw state dict to the child
                ds.load_state_dict(child_states[name]) 