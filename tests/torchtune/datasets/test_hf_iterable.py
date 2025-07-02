# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import islice
from pathlib import Path

import pytest

from torchdata.stateful_dataloader import StatefulDataLoader

from torchtune.data.metrics import MetricsAggregator, StandardMetricTransform
from torchtune.datasets import HfIterableDataset

from .test_iterable_utils import collate_with_metrics, generate_ckpt

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
                f'{{"id": {sample_id}, "tokens": {tokens}, "text": "sample_{sample_id}", "labels": {tokens}}}\n'
            )


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
        epoch_samples = list(islice(iter(unshuffled_ds), SMALL_DATASET_SIZE * 2))

        first_epoch_samples = epoch_samples[:SMALL_DATASET_SIZE]
        second_epoch_samples = epoch_samples[SMALL_DATASET_SIZE:]

        # Unshuffled should have same order in both epochs
        first_epoch_ids = [sample["id"] for sample in first_epoch_samples]
        second_epoch_ids = [sample["id"] for sample in second_epoch_samples]
        assert first_epoch_ids == list(range(SMALL_DATASET_SIZE))
        assert second_epoch_ids == list(range(SMALL_DATASET_SIZE))

        # Test shuffled dataset
        shuffled_ds = dataset_factory(
            small_dataset_file, dataset_name="shuffled", shuffle=True
        )

        # Collect full epochs to compare
        epoch_samples = list(islice(iter(shuffled_ds), SMALL_DATASET_SIZE * 2))

        first_epoch_samples = epoch_samples[:SMALL_DATASET_SIZE]
        second_epoch_samples = epoch_samples[SMALL_DATASET_SIZE:]

        # Shuffled epochs should have different order
        first_epoch_ids = [sample["id"] for sample in first_epoch_samples]
        second_epoch_ids = [sample["id"] for sample in second_epoch_samples]
        assert first_epoch_ids != list(
            range(SMALL_DATASET_SIZE)
        ), f"Shuffled should not be sorted, got {first_epoch_ids}"
        assert (
            first_epoch_ids != second_epoch_ids
        ), f"Shuffled epochs should be shuffled differently, got {first_epoch_ids} and {second_epoch_ids}"

        # But should contain the same set of IDs
        assert set(first_epoch_ids) == set(
            range(SMALL_DATASET_SIZE)
        ), f"First epoch samples should be (0-{SMALL_DATASET_SIZE-1}), got {first_epoch_ids}"
        assert set(second_epoch_ids) == set(
            range(SMALL_DATASET_SIZE)
        ), f"Second epoch samples should be (0-{SMALL_DATASET_SIZE-1}), got {second_epoch_ids}"

    def test_epoch_tracking(self, dataset_factory, small_dataset_file):
        """Test that epoch number is correctly tracked across dataset restarts."""
        dataset = dataset_factory(small_dataset_file, shuffle=False)

        # Two epoch samples
        epoch_samples = list(islice(iter(dataset), SMALL_DATASET_SIZE * 2))

        first_epoch_samples = epoch_samples[:SMALL_DATASET_SIZE]
        second_epoch_samples = epoch_samples[SMALL_DATASET_SIZE:]

        # All should have epoch 0
        first_epoch_metrics = []
        for sample in first_epoch_samples:
            first_epoch_metrics.extend(sample["metrics"])
        epoch_values = [
            metric.value for metric in first_epoch_metrics if metric.name == "epoch"
        ]
        assert all(
            epoch_value == 0 for epoch_value in epoch_values
        ), f"Epoch values should be 0, got {epoch_values}"

        # All should have epoch 1
        second_epoch_metrics = []
        for sample in second_epoch_samples:
            second_epoch_metrics.extend(sample["metrics"])
        epoch_values = [
            metric.value for metric in second_epoch_metrics if metric.name == "epoch"
        ]
        assert all(
            epoch_value == 1 for epoch_value in epoch_values
        ), f"Epoch values should be 1, got {epoch_values}"
