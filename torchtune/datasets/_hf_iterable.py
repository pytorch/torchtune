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
    MetricTransform,
)
from torchtune.datasets._iterable_base import DatasetInfo, InfiniteTuneIterableDataset

logger = logging.getLogger(__name__)


class HfIterableDataset(InfiniteTuneIterableDataset):
    """HuggingFace dataset with infinite iteration and composable transforms.

    This is an infinite dataset that wraps a HuggingFace dataset. After exhausting
    the dataset, it will restart from the beginning.

    Transform pipeline: raw_data -> message_transform -> model_transform -> output_transform -> metric_transform

    This dataset is responsible for:
      - Loading and sharding the dataset
      - Shuffling at initialization and after each epoch
      - Applying transforms to the data
      - Returning an infinite iterator over the dataset

    Args:
        message_transform (Optional[Callable]): Transforms raw data into a `Message`.
        model_transform (Optional[Callable]): Prepares messages for the model,
            usually by tokenizing them.
        output_transform (Optional[Callable]): Prepares tokenized inputs for the
            recipe, often by manipulating labels (e.g., setting an ignore index).
            This transform is recipe-dependent (e.g., SFT, DPO, etc.).
        metric_transform (Optional[MetricTransform]): Computes metrics from a
            sample (e.g., token count). If ``None``, a default transform is used.
            To disable standard metric tracking, set this to ``lambda x: x``.
        shuffle_buffer_size (Optional[int]): Size of the shuffle buffer.
            If ``None`` or 0, no shuffling is performed.
        weight (Optional[float]): Weight for this dataset. Defaults to 1.0.
        seed (int): Seed for shuffling.
        num_shards_per_rank (int): The target number of shards per worker (GPU).
            The actual number of shards will be a multiple of
            ``world_size * dataloader_workers``.
        dataset_name (Optional[str]): Name of the dataset. If ``None``, a name is
            generated from the ``path``, ``source``, and ``split``.
        filter_fn (Optional[Callable]): A function to filter the dataset.
        filter_kwargs (Optional[dict[str, Any]]): Keyword arguments for ``filter_fn``.
        **load_dataset_kwargs: Keyword arguments for the
            :func:`~datasets.load_dataset` function.
    """

    def __init__(
        self,
        *,
        message_transform: Optional[Callable] = None,
        model_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        metric_transform: Optional[MetricTransform] = None,
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
        self._weight = weight if weight is not None else 1.0

        # Create default transform if not provided
        self._metric_transform = metric_transform or DefaultTrainingMetricTransform()

        # Auto-generate dataset name if not provided, ensuring it's always a string
        if dataset_name is None:
            path = load_dataset_kwargs.get("path", None)
            source = load_dataset_kwargs.get("source", None)
            split = load_dataset_kwargs.get("split", None)
            name_parts = []
            for item in [path, source, split]:
                if item is not None:
                    name_parts.append(str(item).replace("/", "_"))
            dataset_name = "_".join(name_parts)

        # Build the hierarchical info object for this dataset
        self._info = DatasetInfo(name=dataset_name, weight=self._weight)

        # Set dataset name on the transform if it supports it
        if hasattr(self._metric_transform, "set_dataset_name"):
            self._metric_transform.set_dataset_name(dataset_name)

        # Internal state for resumption
        self._num_epochs = 0

        # Load and setup HF dataset
        self._setup_hf_dataset(
            load_dataset_kwargs, num_shards_per_rank, filter_fn, filter_kwargs
        )

    @property
    def info(self) -> DatasetInfo:
        """Returns info for this leaf dataset, which has no children."""
        return self._info

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
        One-time setup of HuggingFace dataset that handles Handles distributed sharding,
        shuffle configuration, and filtering.

        Called once during __init__ to avoid expensive re-computation.
        """

        # Distributed setup
        world_size, rank = 1, 0
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

        # Load and shard dataset
        ds = load_dataset(**load_dataset_kwargs)

        # Use to_iterable_dataset for non-streaming datasets
        is_streaming = load_dataset_kwargs.get("streaming", False)
        if is_streaming:
            logger.warning(
                f"Streaming datasets were not yet tested for distributed training. "
                f"split_dataset_by_node is applied, but no resharding was done manually. "
                f"Dataset '{self.info.name}' has "
                f"{getattr(ds, 'num_shards', 'unknown')}, and your training has {world_size} ranks."
                f"See: https://huggingface.co/docs/datasets/en/package_reference/main_classes?#datasets.IterableDataset.shard"
                f"Consider setting streaming=False, which should also be faster."
            )
        if not is_streaming:
            # Define number of shards based on (world_size, num of shards per GPU, dataloader workers)
            # E.g. world_size=2, num_shards_per_rank=16, dataloader_workers=3
            # we will try 2*16 = 32 shards. Since 32 is not a multiple of 6, we will do 36 shards.
            # Each rank gets 18 shards, each dataloader worker in that rank gets 6 shards.
            worker_info = torch.utils.data.get_worker_info()
            num_dataloader_workers = worker_info.num_workers if worker_info else 1

            # Calculate total workers across all ranks and dataloader processes
            total_workers = world_size * num_dataloader_workers

            # Find minimum shards that satisfies our target while being divisible by workers
            desired_shards = world_size * num_shards_per_rank

            # Round up to next multiple of total_workers for even distribution
            if desired_shards % total_workers == 0:
                num_shards = desired_shards
            else:
                num_shards = total_workers * (
                    (desired_shards + total_workers - 1) // total_workers
                )

            # If the dataset is not streaming and has a defined length,
            # we cannot have num_shards > dataset_size.
            if hasattr(ds, "__len__"):
                dataset_size = len(ds)
                if num_shards > dataset_size:
                    raise ValueError(
                        f"Number of shards ({num_shards}) is greater than the dataset size ({dataset_size})."
                        f"Please decrease one of {num_shards_per_rank=} or {num_dataloader_workers=} or {world_size=}."
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
        """Infinite iteration over dataset samples.

        Behavior:
        - Restarts from beginning when dataset is exhausted
        - Reshuffles at start of each epoch (if enabled)
        - Applies full transform pipeline to each sample
        - Adds 'num_epochs' metric to track dataset progress
        - Yields samples indefinitely for continuous training
        """

        while True:  # Infinite iteration
            self._ds.set_epoch(self._num_epochs)
            epoch_iterator = iter(self._ds)
            samples_yielded = 0

            try:
                for sample in epoch_iterator:
                    # NOTE: We apply transforms here instead of using .map() to work around
                    # HuggingFace datasets bug where .map() causes incorrect checkpoint resumption.
                    # See: https://github.com/huggingface/datasets/issues/7630
                    # This ensures transforms are applied fresh on each sample during iteration.
                    sample = self._apply_transforms(sample)

                    # Track the number of epochs completed for each dataset. This is
                    # especially useful when interleaving multiple datasets, but
                    # also necessary to track dataset-level metrics.
                    metric_num_epochs = Metric(
                        dataset_name=self.info.name,
                        metric_name="num_epochs",
                        value=self._num_epochs,
                        agg_type=AggregationType.MAX,
                    )
                    if "metrics" not in sample:
                        sample["metrics"] = []
                    sample["metrics"].append(metric_num_epochs)

                    samples_yielded += 1
                    yield sample

            except StopIteration:
                # Expected when dataset is exhausted
                pass
            except Exception as e:
                logger.error(
                    f"Dataset {self.info.name} encountered an unexpected error: {e}."
                )
                raise

            # Check if we got zero samples - this might indicate an issue
            if samples_yielded == 0:
                logger.warning(
                    f"Dataset {self.info.name} epoch {self._num_epochs} yielded 0 samples - potential issue!"
                )

            # Epoch complete - increment and continue infinite loop
            self._num_epochs += 1

    def state_dict(self) -> dict[str, Any]:
        """Returns dataset checkpoint state."""
        hf_state = self._ds.state_dict()
        state = {
            "num_epochs": self._num_epochs,
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
