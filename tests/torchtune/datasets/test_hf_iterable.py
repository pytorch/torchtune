# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for HfIterableDataset core functionality.

This module tests the foundational iterable dataset capabilities including:
- Basic iteration and data loading
- Epoch boundary handling and tracking
- Shuffling behavior across epochs
- Checkpointing and state restoration
- Distributed training scenarios

Uses synthetic JSON data with predictable patterns to verify correct behavior.
"""

import math
import shutil
import tempfile
from itertools import islice
from pathlib import Path

import pytest
import torch.distributed as dist
from tests.test_utils import gpu_test
from torch.testing._internal.common_fsdp import FSDPTest

from torchdata.stateful_dataloader import StatefulDataLoader

from torchtune.data.metrics import DefaultTrainingMetricTransform, MetricsAggregator
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
            metric_transform=DefaultTrainingMetricTransform(),
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
            metric_transform=DefaultTrainingMetricTransform(),
            num_shards_per_rank=4,
        )

        # Should generate name from path and split
        assert dataset.info.name == "json_train"
        # Test default sampling weight
        assert dataset.info.weight == 1.0

        # Test giving a name and custom weight
        custom_weight = 2.5
        dataset2 = HfIterableDataset(
            path="json",
            data_files=small_dataset_file,
            split="train",
            dataset_name="my_dataset",
            weight=custom_weight,
            seed=SEED,
            metric_transform=DefaultTrainingMetricTransform(),
            num_shards_per_rank=4,
        )

        # Should use provided name and weight
        assert dataset2.info.name == "my_dataset"
        # Test custom sampling weight
        assert dataset2.info.weight == custom_weight

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

        # Extract IDs for comparison
        first_epoch_ids = [sample["id"] for sample in first_epoch_samples]
        second_epoch_ids = [sample["id"] for sample in second_epoch_samples]

        # Shuffled epochs should have different order
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
            metric.value
            for metric in first_epoch_metrics
            if metric.metric_name == "epoch"
        ]
        assert all(
            epoch_value == 0 for epoch_value in epoch_values
        ), f"Epoch values should be 0, got {epoch_values}"

        # All should have epoch 1
        second_epoch_metrics = []
        for sample in second_epoch_samples:
            second_epoch_metrics.extend(sample["metrics"])
        epoch_values = [
            metric.value
            for metric in second_epoch_metrics
            if metric.metric_name == "epoch"
        ]
        assert all(
            epoch_value == 1 for epoch_value in epoch_values
        ), f"Epoch values should be 1, got {epoch_values}"


class TestDistributedHfIterableDataset(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @gpu_test(gpu_count=2)
    def test_distributed_epoch_boundary_checkpointing(self):
        """
        Test epoch boundary handling with checkpointing in distributed setting.
        Ensures proper handling of:
        - Checkpointing at 0.9, 1.0, and 2.5 epoch boundaries
        - Correct sample distribution across epochs
        - Proper state restoration after checkpointing
        """
        rank = dist.get_rank()

        # Create shared temp directory (only rank 0 creates it)
        if rank == 0:
            temp_dir = tempfile.mkdtemp(prefix="epoch_test_")
        else:
            temp_dir = ""

        # Broadcast temp directory path to all ranks
        temp_dir_list = [temp_dir]
        dist.broadcast_object_list(temp_dir_list, src=0)
        temp_dir = temp_dir_list[0]
        tmp_path = Path(temp_dir)

        try:
            medium_dataset_file = tmp_path / "medium_data.json"

            # Only rank 0 creates the data file, all ranks read from it
            if rank == 0:
                create_test_json_file(medium_dataset_file, MEDIUM_DATASET_SIZE)
            dist.barrier()  # Wait for file creation

            # Test multiple epoch boundaries
            for num_epochs in [0.9, 1.0, 2.5]:

                def create_loader_and_aggregator():
                    dataset = HfIterableDataset(
                        path="json",
                        data_files=str(medium_dataset_file),
                        split="train",
                        dataset_name="epoch_test",
                        seed=SEED,
                        shuffle_buffer_size=0,  # No shuffle for determinism
                        metric_transform=DefaultTrainingMetricTransform(),
                        num_shards_per_rank=2,
                    )
                    loader = StatefulDataLoader(
                        dataset,
                        batch_size=BATCH_SIZE,
                        collate_fn=collate_with_metrics,
                        num_workers=0,
                    )
                    return loader, MetricsAggregator()

                loader1, aggregator1 = create_loader_and_aggregator()
                loader2, aggregator2 = create_loader_and_aggregator()

                # Calculate steps to reach desired epoch boundary
                samples_per_rank = MEDIUM_DATASET_SIZE // dist.get_world_size()
                total_samples = int(samples_per_rank * num_epochs)
                total_steps = total_samples // BATCH_SIZE

                if total_steps < 2:
                    raise ValueError(
                        f"Not enough steps for meaningful test: {total_steps}"
                    )

                # Split steps between before and after checkpoint
                steps_before = max(1, total_steps // 2)
                steps_after = total_steps - steps_before

                result = generate_ckpt(
                    loader1,
                    aggregator1,
                    steps_before,
                    steps_after,
                    resume_dataloader=loader2,
                    resume_aggregator=aggregator2,
                )

                # Verify deterministic resumption - critical for distributed training
                orig_post_ids = [
                    b["id"].tolist() for b in result["post_checkpoint_batches"]
                ]
                resumed_ids = [b["id"].tolist() for b in result["resumed_batches"]]
                assert orig_post_ids == resumed_ids, (
                    f"Rank {rank}: Non-deterministic resume for {num_epochs} epochs. "
                    f"This indicates checkpoint/resume state is not properly preserved."
                )

                # Verify epoch metric is correctly tracked
                final_metrics = result["final_metrics"]
                expected_epoch = math.floor(
                    num_epochs - 1e-9
                )  # -1e-9 so 1.0 epochs -> 0
                assert (
                    final_metrics["train_epoch_test/num_epochs"] == expected_epoch
                ), f"Epoch count incorrect for {num_epochs} epochs test scenario"

        finally:
            # Clean up temp directory (only rank 0)
            if rank == 0:
                shutil.rmtree(temp_dir)
