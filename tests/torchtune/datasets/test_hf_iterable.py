# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import islice
from pathlib import Path
from typing import Any, Optional

import pytest
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtune.data import (
    MetricsAggregator,
    padded_collate_sft,
    StandardMetricTransform,
)
from torchtune.datasets import HfIterableDataset


# Test Constants - Avoid perfect divisions
SMALL_DATASET_SIZE = 23
MEDIUM_DATASET_SIZE = 35
SEED = 42
BATCH_SIZE = 5
DEFAULT_SHUFFLE_BUFFER_SIZE = 8


def create_test_json_file(path: Path, num_samples: int, offset: int = 0) -> None:
    """Creates a dummy JSON test data file with token samples of varying lengths.

    Args:
        path (Path): The path to the file to create
        num_samples (int): The number of samples to create
        offset (int): The offset to add to the sample ID to ensure unique IDs in different datasets
    """
    with open(path, "w") as f:
        for i in range(num_samples):
            sample_id = i + offset
            # Realistic token length variation (1-3 tokens)
            token_len = (i % 3) + 1
            tokens = list(range(sample_id, sample_id + token_len))
            f.write(
                f'{{"id": {sample_id}, "tokens": {tokens}, "text": "sample_{sample_id}"}}\n'
            )


def collate_with_metrics(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function that extracts metrics and uses padded_collate_sft as base collator."""
    # Extract metrics first
    all_metrics = []
    clean_batch = []
    for sample in batch:
        if "metrics" in sample:
            all_metrics.extend(sample.pop("metrics"))
        clean_batch.append(sample)

    if not clean_batch:
        return {"metrics": all_metrics}

    # Use torchtune's padded_collate_sft as base collator
    collated_batch = padded_collate_sft(clean_batch)
    collated_batch["metrics"] = all_metrics
    return collated_batch


def generate_ckpt(
    dataloader: StatefulDataLoader,
    aggregator: MetricsAggregator,
    steps_before_checkpoint: int,
    steps_after_checkpoint: int,
    resume_dataloader: Optional[StatefulDataLoader] = None,
    resume_aggregator: Optional[MetricsAggregator] = None,
) -> dict[str, Any]:
    """
    Generates a checkpoint by running through data and saving checkpoint mid-stream.
    Optionally, a second dataloader and aggregator can be given to resume from ckpt
    and run steps_after_checkpoint to match the first one.

    Args:
        dataloader (StatefulDataLoader): The dataloader to test
        aggregator (MetricsAggregator): The metrics aggregator to use
        steps_before_checkpoint (int): Number of steps to run before saving checkpoint
        steps_after_checkpoint (int): Number of steps to run after checkpoint
        resume_dataloader (Optional[StatefulDataLoader]): Optional new dataloader to test resuming.
            If None, returns empty resumed_batches.
        resume_aggregator (Optional[MetricsAggregator]): Optional new aggregator to test resuming.
            If None, returns empty resumed_metrics.

    Returns:
        dict[str, Any]: Dict with batches/metrics from both pre and post checkpoint runs.
    """
    iterator = iter(dataloader)

    # Collect batches before and after checkpoint
    batches = []
    checkpoint_state = None
    metrics_at_checkpoint = {}

    total_steps = steps_before_checkpoint + steps_after_checkpoint

    for idx, batch in enumerate(iterator):
        batches.append(batch)

        # Process metrics
        if "metrics" in batch:
            aggregator.update(batch.pop("metrics"))

        # Save checkpoint state after steps_before_checkpoint
        if idx == steps_before_checkpoint - 1:  # -1 because idx is 0-based
            checkpoint_state = {
                "loader": dataloader.state_dict(),
                "aggregator": aggregator.state_dict(),
            }
            metrics_at_checkpoint = aggregator.get_metrics_for_logging(prefix="train")

        # Stop after total steps
        if idx == total_steps - 1:
            break

    # Split batches
    pre_checkpoint_batches = batches[:steps_before_checkpoint]
    post_checkpoint_batches = batches[steps_before_checkpoint:]

    # Resume with new instances if provided
    resumed_batches = []
    resumed_metrics = {}

    if (
        resume_dataloader is not None
        and resume_aggregator is not None
        and checkpoint_state is not None
    ):
        # Test resuming with new instances
        resume_dataloader.load_state_dict(checkpoint_state["loader"])
        resume_aggregator.load_state_dict(checkpoint_state["aggregator"])
        resume_iterator = iter(resume_dataloader)

        # Collect only the post-checkpoint batches when resuming
        for idx, batch in enumerate(resume_iterator):
            resumed_batches.append(batch)

            # Process metrics
            if "metrics" in batch:
                resume_aggregator.update(batch.pop("metrics"))

            # Stop after steps_after_checkpoint
            if idx == steps_after_checkpoint - 1:
                break

        resumed_metrics = resume_aggregator.get_metrics_for_logging(prefix="train")

    return {
        # Original run
        "pre_checkpoint_batches": pre_checkpoint_batches,
        "post_checkpoint_batches": post_checkpoint_batches,
        "metrics_at_checkpoint": metrics_at_checkpoint,
        "final_metrics": aggregator.get_metrics_for_logging(prefix="train"),
        # Resumed run
        "resumed_batches": resumed_batches,
        "resumed_metrics": resumed_metrics,
        # Internal state for loading - only if someone needs to manually load
        "_checkpoint_state": checkpoint_state,
    }


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide temporary directory for test data files."""
    return tmp_path


@pytest.fixture
def small_dataset_file(tmp_data_dir):
    path = tmp_data_dir / "small_data.json"
    create_test_json_file(path, SMALL_DATASET_SIZE, offset=0)
    return str(path)


@pytest.fixture
def dataset_factory():
    """Factory for creating HfIterableDataset instances with common defaults."""

    def _create_dataset(
        data_file: str,
        dataset_name: str = "test_dataset",
        shuffle: bool = False,
        **kwargs,
    ) -> HfIterableDataset:
        return HfIterableDataset(
            path="json",
            data_files=data_file,
            split="train",
            dataset_name=dataset_name,
            seed=SEED,
            shuffle_buffer_size=10 if shuffle else 0,
            metric_transform=StandardMetricTransform(),
            num_shards_per_rank=2,
            **kwargs,
        )

    return _create_dataset


class TestHfIterableDataset:
    """Tests for HfIterableDataset basic functionality."""

    def test_default_dataset_name(self, small_dataset_file):
        """Test that dataset name is auto-generated from path when not provided."""
        # Create dataset without specifying name
        dataset = HfIterableDataset(
            path="json",
            data_files=small_dataset_file,
            split="train",
            # dataset_name not provided - should auto-generate
            seed=SEED,
            metric_transform=StandardMetricTransform(),
            num_shards_per_rank=4,
        )

        # Should generate name from path and split
        assert dataset.dataset_name == "json_train"

        # Test giving a name
        dataset2 = HfIterableDataset(
            path="json",
            data_files=small_dataset_file,
            split="train",
            dataset_name="my_dataset",
            seed=SEED,
            metric_transform=StandardMetricTransform(),
            num_shards_per_rank=4,
        )

        # Should generate name from path and split
        assert dataset2.dataset_name == "my_dataset"

    @pytest.mark.parametrize("num_epochs", [0.5, 1.0, 2.5])
    def test_epoch_boundaries_and_checkpointing(
        self, num_epochs, dataset_factory, small_dataset_file
    ):
        """
        Tests that for N epochs, each sample appears exactly N times (rounded down),
        the epoch metric is correct, and checkpointing works as expected.
        """

        # 1. Setup Dataloaders and Aggregators for original and resumed runs
        def create_loader_and_aggregator():
            dataset = dataset_factory(small_dataset_file, shuffle=False)
            loader = StatefulDataLoader(
                dataset, batch_size=BATCH_SIZE, collate_fn=collate_with_metrics
            )
            aggregator = MetricsAggregator()
            return loader, aggregator

        loader1, aggregator1 = create_loader_and_aggregator()
        loader2, aggregator2 = create_loader_and_aggregator()

        # 2. Calculate steps for the test run
        total_samples = int(SMALL_DATASET_SIZE * num_epochs)
        total_steps = total_samples // BATCH_SIZE

        steps_before_checkpoint = max(1, total_steps // 2)
        steps_after_checkpoint = total_steps - steps_before_checkpoint

        # 3. Generate checkpoint and resume
        result = generate_ckpt(
            loader1,
            aggregator1,
            steps_before_checkpoint=steps_before_checkpoint,
            steps_after_checkpoint=steps_after_checkpoint,
            resume_dataloader=loader2,
            resume_aggregator=aggregator2,
        )

        # 4. Verify checkpointing and resumption
        orig_post_ids = [b["id"].tolist() for b in result["post_checkpoint_batches"]]
        resumed_ids = [b["id"].tolist() for b in result["resumed_batches"]]
        assert (
            orig_post_ids == resumed_ids
        ), "Resumed batches should be identical for deterministic run"
        assert (
            result["final_metrics"] == result["resumed_metrics"]
        ), "Final metrics should match"

    def test_shuffling_behavior(self, dataset_factory, small_dataset_file):
        """Tests that shuffling changes data order between epochs but preserves the set of samples."""
        # Test unshuffled dataset
        unshuffled_ds = dataset_factory(
            small_dataset_file, dataset_name="unshuffled", shuffle=False
        )

        # Get samples from two passes through the dataset
        epoch_samples = islice(iter(unshuffled_ds), SMALL_DATASET_SIZE * 2)

        first_epoch_samples = epoch_samples[:SMALL_DATASET_SIZE]
        second_epoch_samples = epoch_samples[SMALL_DATASET_SIZE:]

        # Unshuffled should have same order in both epochs
        assert first_epoch_samples == list(range(SMALL_DATASET_SIZE))
        assert second_epoch_samples == list(range(SMALL_DATASET_SIZE))

        # Test shuffled dataset
        shuffled_ds = dataset_factory(
            small_dataset_file, dataset_name="shuffled", shuffle=True
        )

        # Collect full epochs to compare
        epoch_samples = islice(iter(shuffled_ds), SMALL_DATASET_SIZE * 2)

        first_epoch_samples = epoch_samples[:SMALL_DATASET_SIZE]
        second_epoch_samples = epoch_samples[SMALL_DATASET_SIZE:]

        # Shuffled epochs should have different order
        assert first_epoch_samples != list(
            range(SMALL_DATASET_SIZE)
        ), f"Shuffled should not be sorted, got {first_epoch_samples}"
        assert (
            first_epoch_samples != second_epoch_samples
        ), f"Shuffled epochs should be shuffled differently, got {first_epoch_samples} and {second_epoch_samples}"

        # But should contain the same set of IDs
        assert set(first_epoch_samples) == set(
            range(SMALL_DATASET_SIZE)
        ), f"First epoch samples should be (0-{SMALL_DATASET_SIZE-1}), got {first_epoch_samples}"
        assert set(second_epoch_samples) == set(
            range(SMALL_DATASET_SIZE)
        ), f"Second epoch samples should be (0-{SMALL_DATASET_SIZE-1}), got {second_epoch_samples}"

    def test_epoch_tracking(self, dataset_factory, small_dataset_file):
        """Test that epoch number is correctly tracked across dataset restarts."""
        dataset = dataset_factory(small_dataset_file, shuffle=False)

        # Two epoch samples
        epoch_samples = islice(iter(dataset), SMALL_DATASET_SIZE * 2)

        first_epoch_samples = epoch_samples[:SMALL_DATASET_SIZE]
        second_epoch_samples = epoch_samples[SMALL_DATASET_SIZE:]

        # All should have epoch 0
        epoch_values = [
            epoch_metric.value for epoch_metric in first_epoch_samples["metrics"]
        ]
        assert all(
            epoch_value == 0 for epoch_value in epoch_values
        ), f"Epoch values should be 0, got {epoch_values}"

        # All should have epoch 1
        epoch_values = [
            epoch_metric.value for epoch_metric in second_epoch_samples["metrics"]
        ]
        assert all(
            epoch_value == 1 for epoch_value in epoch_values
        ), f"Epoch values should be 1, got {epoch_values}"
