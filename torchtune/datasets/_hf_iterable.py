# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Callable, Iterator, Optional

import torch
import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

from torchtune.data.metrics import (
    AggregationType,
    DefaultTrainingMetricTransform,
    Metric,
)
from torchtune.datasets._iterable_base import TuneIterableDataset

logger = logging.getLogger(__name__)


class HfIterableDataset(TuneIterableDataset):
    """HuggingFace dataset implementation with composable metrics.

    This is an infinite dataset. After exhausting the dataset, it will restart from the beginning.

    This dataset is responsible for:
      - Loading and sharding the dataset
      - Shuffling at initialization and after each epoch
      - Applying transforms
      - Returning an infinite iterator over the dataset

      Args:
        message_transform (Optional[Callable]): Transforms raw data into Message
        model_transform (Optional[Callable]): Take messages and prepares it for the model. Usually the tokenizer.
        output_transform (Optional[Callable]): Takes tokenized inputs and prepares it for the recipe. Usually
            does some label manipulation, e.g. ignore index. Think of it as recipe-dependent, e.g. SFT, RL, DPO, etc.
        metric_transform (Optional[Callable]): Takes the sample and computes metrics, e.g. token count.
            If None, a default transform is used. To stop tracking metrics, set it to lambda x: x.
        shuffle_buffer_size (Optional[int]): Size of the shuffle buffer. If None or 0, no shuffling is done.
        seed (int): Seed for shuffling.
        num_shards_per_rank (int): Target number of shards per worker (GPU). It will find a multiple
            of world_size * dataloader_workers.
        dataset_name (Optional[str]): Name of the dataset. If None, a default name is generated
            from the path, source, and split.
        filter_fn (Optional[Callable]): Filter function to apply to the dataset.
        filter_kwargs (Optional[dict[str, Any]]): Keyword arguments to pass to the filter function.
        load_dataset_kwargs (dict[str, Any]): Keyword arguments to pass to the load_dataset function.

    """

    def __init__(
        self,
        *,
        message_transform: Optional[Callable] = None,
        model_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        metric_transform: Optional[Callable] = None,
        shuffle_buffer_size: Optional[int] = 1000,
        weight: Optional[float] = 1.0,
        seed: int = 42,
        num_shards_per_rank: int = 64,
        dataset_name: Optional[str] = None,
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[dict[str, Any]] = None,
        **load_dataset_kwargs,
    ):
        # Store configuration
        self._shuffle_buffer_size = shuffle_buffer_size
        self._seed = seed
        self._message_transform = message_transform
        self._model_transform = model_transform
        self._output_transform = output_transform
        self._weight = weight  # TODO: make it a property?

        # Create default transform if not provided
        self._metric_transform = metric_transform or DefaultTrainingMetricTransform()

        # Auto-generate dataset name if not provided, ensuring it's always a string.
        if dataset_name is None:
            path = load_dataset_kwargs.get("path", None)
            source = load_dataset_kwargs.get("source", None)
            split = load_dataset_kwargs.get("split", None)
            name_parts = []
            for item in [path, source, split]:
                if item is not None:
                    name_parts.append(str(item).replace("/", "_"))
            self._dataset_name: str = "_".join(name_parts)
        else:
            self._dataset_name: str = dataset_name

        # Set dataset name on the transform if it supports it
        if hasattr(self._metric_transform, "set_dataset_name"):
            self._metric_transform.set_dataset_name(self._dataset_name)

        # Internal state for resumption
        self._num_epochs = 0

        # Load and setup HF dataset
        self._setup_hf_dataset(
            load_dataset_kwargs, num_shards_per_rank, filter_fn, filter_kwargs
        )

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def sampling_weight(self) -> float:
        return self._weight

    def _apply_transforms(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply transforms if they exist, otherwise return sample unchanged."""
        if self._message_transform is not None:
            sample = self._message_transform(sample)
        if self._model_transform is not None:
            sample = self._model_transform(sample)
        if self._output_transform is not None:
            sample = self._output_transform(sample)
        if self._metric_transform is not None:
            sample = self._metric_transform(sample)
        return sample

    def _setup_hf_dataset(
        self,
        load_dataset_kwargs: dict[str, Any],
        num_shards_per_rank: int,
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Configures the Hugging Face dataset, including sharding, filtering, and
        transform mapping. This method is called only once during initialization
        to avoid expensive re-computation on each epoch.
        """

        # Distributed setup
        world_size, rank = 1, 0
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

        # Load and shard dataset
        ds = load_dataset(**load_dataset_kwargs)

        # Use to_iterable_dataset for streaming datasets
        if not load_dataset_kwargs.get("streaming", False):

            # Define number of shards based on (world_size, num of shards per GPU, dataloader workers)
            # E.g. world_size=2, num_shards_per_rank=16, dataloader_workers=3
            # we will try 2*16 = 32 shards. Since 32 is not a multiple of 3, we will do 36 shards.
            # Each rank gets 16 shards, each dataloader worker in that rankgets 6 shards.
            worker_info = torch.utils.data.get_worker_info()
            num_dataloader_workers = worker_info.num_workers if worker_info else 1

            # Calculate total workers
            total_workers = world_size * num_dataloader_workers

            # Calculate desired shards
            desired_shards = world_size * num_shards_per_rank

            # Find the smallest multiple of total_workers that is >= desired_shards
            if desired_shards % total_workers == 0:
                num_shards = desired_shards
            else:
                num_shards = total_workers * (
                    (desired_shards + total_workers - 1) // total_workers
                )

            # If the dataset is not streaming and has a defined length,
            # we cannot have num_shards > dataset_size.
            if not load_dataset_kwargs.get("streaming", False) and hasattr(
                ds, "__len__"
            ):
                dataset_size = len(ds)
                if num_shards > dataset_size:
                    raise ValueError(
                        f"Number of shards ({num_shards}) is greater than the dataset size ({dataset_size})."
                        f"Please decrease num_shards_per_rank."
                    )

            ds = ds.to_iterable_dataset(num_shards=num_shards)

        # Shuffle the dataset
        if self._shuffle_buffer_size and self._shuffle_buffer_size > 0:
            ds = ds.shuffle(seed=self._seed, buffer_size=self._shuffle_buffer_size)

        # Distribute across ranks
        if world_size > 1:
            ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)

        # Apply filtering if specified
        if filter_fn:
            filter_kwargs = filter_kwargs or {}
            ds = ds.filter(filter_fn, **filter_kwargs)

        self._ds = ds

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate through the dataset infinitely.

        It will restart from the beginning after exhausting the dataset.

        If shuffle_buffer_size is set, it will shuffle the dataset at the beginning of each epoch
        when set_epoch is called.

        An additional metric "num_epochs" is added to the sample.
        """

        while True:  # Infinite iteration
            epoch_seed = self._seed + self._num_epochs
            self._ds.set_epoch(epoch_seed)
            epoch_iterator = iter(self._ds)
            samples_yielded = 0

            try:
                for sample in epoch_iterator:
                    # NOTE: We apply transforms here instead of using .map() call
                    # to work around https://github.com/huggingface/datasets/issues/7630
                    # where .map() can cause incorrect resumption from a checkpoint.
                    sample = self._apply_transforms(sample)

                    # Track the number of epochs completed for each dataset. This is
                    # especially useful when interleaving multiple datasets, but
                    # also necessary to track dataset-level metrics.
                    metric_num_epochs = Metric(
                        dataset_name=self.dataset_name,
                        name="num_epochs",
                        value=self._num_epochs,
                        agg_type=AggregationType.MAX,
                    )
                    if "metrics" not in sample:
                        sample["metrics"] = []
                    sample["metrics"].append(metric_num_epochs)

                    samples_yielded += 1
                    yield sample

            except StopIteration:
                pass  # Iterator is exhausted, which is expected.
            except Exception as e:
                logger.warning(
                    f"Dataset {self.dataset_name} encountered an unexpected error: {e}."
                )
                raise

            # Check if we got zero samples - this might indicate an issue
            if samples_yielded == 0:
                logger.warning(
                    f"Dataset {self.dataset_name} epoch {self._num_epochs} yielded 0 samples - potential issue!"
                )

            # Epoch complete - increment and continue infinite loop
            self._num_epochs += 1

    def state_dict(self) -> dict[str, Any]:
        """
        The dataset returns its own state directly, without namespacing.
        """
        hf_state = self._ds.state_dict()
        state = {
            "num_epochs": self._num_epochs,
            "seed": self._seed,
            "hf_dataset_state": hf_state,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load state from checkpoint, including restoring the state of the
        Hugging Face IterableDataset.
        """
        self._num_epochs = state_dict["num_epochs"]
        hf_state = state_dict["hf_dataset_state"]

        # HF is responsible for resuming the dataset state
        # where it last left off
        self._ds.load_state_dict(hf_state)
