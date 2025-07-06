# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import tempfile
from itertools import islice
from pathlib import Path
from unittest.mock import patch

import pytest

import torch
import torch.distributed as dist
from tests.test_utils import gpu_test
from torch.testing._internal.common_fsdp import FSDPTest
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtune.data.metrics import DefaultTrainingMetricTransform, MetricsAggregator
from torchtune.datasets import HfIterableDataset, InterleavedDataset

# Import test utilities
from .test_iterable_utils import collate_with_metrics, generate_ckpt

# Test Constants
SMALL_DATASET_SIZE = 23
MEDIUM_DATASET_SIZE = 35
LARGE_DATASET_SIZE = 47
SEED = 42
BATCH_SIZE = 5


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
def medium_dataset_file(tmp_data_dir):
    path = tmp_data_dir / "medium_data.json"
    create_test_json_file(path, MEDIUM_DATASET_SIZE, offset=100)
    return str(path)


@pytest.fixture
def large_dataset_file(tmp_data_dir):
    path = tmp_data_dir / "large_data.json"
    create_test_json_file(path, LARGE_DATASET_SIZE, offset=1000)
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


class TestInterleavedDataset:
    """Tests for multi-dataset interleaving functionality."""

    def test_initialization_validation(self, dataset_factory, small_dataset_file):
        """Tests that the dataset raises errors for invalid configurations, like duplicate names."""

        # Test 1: Duplicate dataset names should raise an error
        ds1 = dataset_factory(small_dataset_file, dataset_name="ds1", weight=0.5)
        ds2 = dataset_factory(small_dataset_file, dataset_name="ds1", weight=0.5)

        with pytest.raises(
            ValueError, match="Duplicate dataset names found in hierarchy"
        ):
            InterleavedDataset(datasets=[ds1, ds2], seed=SEED)

        # Test 2: Nested interleaved datasets should be supported
        ds3 = dataset_factory(small_dataset_file, dataset_name="ds3", weight=1.5)
        interleaved_child = InterleavedDataset(
            [ds1, ds3], seed=SEED, dataset_name="interleaved_child"
        )

        # Create a parent interleaved dataset containing the nested one
        ds4 = dataset_factory(small_dataset_file, dataset_name="ds4", weight=0.5)

        # Test 3: Weight normalization should work with a warning
        with patch("logging.Logger.warning") as mock_warning:
            interleaved_parent = InterleavedDataset(
                [interleaved_child, ds4], seed=SEED, dataset_name="interleaved_parent"
            )

            # Verify that a warning was logged about weight normalization
            mock_warning.assert_called_once()
            warning_message = mock_warning.call_args[0][0]
            assert "normalized" in warning_message.lower()

            # Verify the hierarchical structure is correct
            assert interleaved_parent.info.name == "interleaved_parent"
            assert len(interleaved_parent.info.children) == 2
            # Datasets are sorted alphabetically, so ds4 comes before interleaved_child
            assert interleaved_parent.info.children[0].name == "ds4"
            assert interleaved_parent.info.children[1].name == "interleaved_child"

            # Verify the nested structure within the nested dataset
            # interleaved_child is at index 1 due to alphabetical sorting
            nested_info = interleaved_parent.info.children[1]
            assert len(nested_info.children) == 2
            assert nested_info.children[0].name == "ds1"
            assert nested_info.children[1].name == "ds3"

            # Verify that sampling weights are normalized to sum to 1.0
            # Access the internal normalized weights tensor
            normalized_weights = interleaved_parent._normalized_weights
            assert isinstance(normalized_weights, torch.Tensor)
            assert len(normalized_weights) == 2

            # ds4: 0.5/(0.5+1.0) = 1/3, interleaved_child: 1.0/(0.5+1.0) = 2/3
            assert abs(normalized_weights[0].item() - 1 / 3) < 1e-3
            assert abs(normalized_weights[1].item() - 2 / 3) < 1e-3
            assert abs(normalized_weights.sum().item() - 1.0) < 1e-6

            # Verify that original weights in info remain unnormalized
            child_weights = [child.weight for child in interleaved_parent.info.children]
            assert abs(child_weights[0] - 0.5) < 1e-6  # ds4 original weight
            assert (
                abs(child_weights[1] - 1.0) < 1e-6
            )  # interleaved_child original weight

    def test_single_dataset(self, dataset_factory, small_dataset_file):
        """Tests that InterleavedDataset works correctly with a single dataset."""
        # Create a single dataset
        ds1 = dataset_factory(small_dataset_file, dataset_name="ds1", weight=0.5)

        # Should work without issues
        interleaved = InterleavedDataset([ds1], seed=SEED)

        # Verify the hierarchical structure
        assert interleaved.info.name == "interleaved_dataset"  # default name
        assert len(interleaved.info.children) == 1
        assert interleaved.info.children[0].name == "ds1"
        assert interleaved.info.children[0].weight == 0.5

        # Verify normalized weights sum to 1.0 (single dataset gets weight 1.0)
        normalized_weights = interleaved._normalized_weights
        assert isinstance(normalized_weights, torch.Tensor)
        assert len(normalized_weights) == 1
        assert abs(normalized_weights[0].item() - 1.0) < 1e-6

        # Test that iteration works correctly
        samples = list(islice(iter(interleaved), 10))
        assert len(samples) == 10

        # All samples should come from the single dataset (ds1 has IDs 0-22)
        sample_ids = {sample["id"] for sample in samples}
        expected_ids = set(range(10))  # ds1 has IDs 0-22
        assert sample_ids == expected_ids

    def test_sampling_ratios(
        self,
        dataset_factory,
        small_dataset_file,
        medium_dataset_file,
        large_dataset_file,
    ):
        """Tests that datasets are sampled according to their assigned weights in nested structure."""
        # Create three datasets with distinct ID ranges
        # ds1 has IDs 0-22, ds2 has IDs 100-134, ds3 has IDs 1000-1046
        ds1 = dataset_factory(small_dataset_file, dataset_name="ds1", weight=0.3)
        ds2 = dataset_factory(medium_dataset_file, dataset_name="ds2", weight=0.7)
        ds3 = dataset_factory(large_dataset_file, dataset_name="ds3", weight=1.0)

        # Create nested structure: interleaved([interleaved([ds1, ds2]), ds3])
        child_interleaved = InterleavedDataset(
            [ds1, ds2], seed=SEED, dataset_name="child"
        )
        parent_interleaved = InterleavedDataset(
            [child_interleaved, ds3], seed=SEED, dataset_name="parent"
        )

        # Collect 400 samples
        sample_count = 400
        samples = list(islice(iter(parent_interleaved), sample_count))

        # Count samples by checking ID ranges
        ds1_count = sum(1 for s in samples if 0 <= s["id"] < SMALL_DATASET_SIZE)
        ds2_count = sum(
            1 for s in samples if 100 <= s["id"] < (MEDIUM_DATASET_SIZE + 100)
        )
        ds3_count = sum(
            1 for s in samples if 1000 <= s["id"] < (LARGE_DATASET_SIZE + 1000)
        )

        assert ds1_count + ds2_count + ds3_count == sample_count

        # Calculate ratios
        ds1_ratio = ds1_count / sample_count
        ds2_ratio = ds2_count / sample_count
        ds3_ratio = ds3_count / sample_count

        # Expected ratios based on nested weighting:
        # Inner weights: ds1=0.3, ds2=0.7 -> inner total=1.0
        # Outer weights: inner=1.0, ds3=1.0 -> normalized to 0.5 each
        # Final ratios: ds1=0.5*0.3=0.15, ds2=0.5*0.7=0.35, ds3=0.5
        expected_ds1_ratio = 0.15
        expected_ds2_ratio = 0.35
        expected_ds3_ratio = 0.5

        # Allow 10% tolerance due to randomness
        assert (
            abs(ds1_ratio - expected_ds1_ratio) < 0.1
        ), f"ds1 ratio {ds1_ratio:.2f} should be ~{expected_ds1_ratio}"
        assert (
            abs(ds2_ratio - expected_ds2_ratio) < 0.1
        ), f"ds2 ratio {ds2_ratio:.2f} should be ~{expected_ds2_ratio}"
        assert (
            abs(ds3_ratio - expected_ds3_ratio) < 0.1
        ), f"ds3 ratio {ds3_ratio:.2f} should be ~{expected_ds3_ratio}"

    def test_metrics_aggregation(
        self,
        dataset_factory,
        small_dataset_file,
        medium_dataset_file,
        large_dataset_file,
    ):
        """Tests that metrics from all child datasets are collected and aggregated in nested structure."""
        ds1 = dataset_factory(small_dataset_file, dataset_name="ds1", weight=0.2)
        ds2 = dataset_factory(medium_dataset_file, dataset_name="ds2", weight=0.8)
        ds3 = dataset_factory(large_dataset_file, dataset_name="ds3", weight=1.0)

        # Create nested structure: interleaved([interleaved([ds1, ds2]), ds3])
        child_interleaved = InterleavedDataset(
            [ds1, ds2], seed=SEED, dataset_name="child"
        )
        parent_interleaved = InterleavedDataset(
            [child_interleaved, ds3], seed=SEED, dataset_name="parent"
        )

        aggregator = MetricsAggregator()

        # Process some samples
        total_samples = 300
        for sample in islice(iter(parent_interleaved), total_samples):
            aggregator.update(sample["metrics"])

        metrics = aggregator.get_metrics_for_logging(prefix="train")

        # Should have metrics from all three datasets, with flat keys
        assert "train_ds1/samples_seen" in metrics
        assert "train_ds2/samples_seen" in metrics
        assert "train_ds3/samples_seen" in metrics

        # All datasets should have contributed samples
        assert metrics["train_ds1/samples_seen"] > 0
        assert metrics["train_ds2/samples_seen"] > 0
        assert metrics["train_ds3/samples_seen"] > 0

        # Total samples should equal what we processed
        calculated_total_samples = (
            metrics["train_ds1/samples_seen"]
            + metrics["train_ds2/samples_seen"]
            + metrics["train_ds3/samples_seen"]
        )
        assert calculated_total_samples == total_samples

        # Test that ratios are approximately correct based on nested weighting
        ds1_ratio = metrics["train_ds1/samples_seen"] / total_samples
        ds2_ratio = metrics["train_ds2/samples_seen"] / total_samples
        ds3_ratio = metrics["train_ds3/samples_seen"] / total_samples

        # Expected ratios based on nested weighting:
        # Inner weights: ds1=0.2, ds2=0.8 -> inner total=1.0
        # Outer weights: inner=1.0, ds3=1.0 -> normalized to 0.5 each
        # Final ratios: ds1=0.5*0.2=0.1, ds2=0.5*0.8=0.4, ds3=0.5
        expected_ds1_ratio = 0.1
        expected_ds2_ratio = 0.4
        expected_ds3_ratio = 0.5

        # Allow 10% tolerance due to randomness
        assert (
            abs(ds1_ratio - expected_ds1_ratio) < 0.1
        ), f"ds1 ratio {ds1_ratio:.2f} should be ~{expected_ds1_ratio}"
        assert (
            abs(ds2_ratio - expected_ds2_ratio) < 0.1
        ), f"ds2 ratio {ds2_ratio:.2f} should be ~{expected_ds2_ratio}"
        assert (
            abs(ds3_ratio - expected_ds3_ratio) < 0.1
        ), f"ds3 ratio {ds3_ratio:.2f} should be ~{expected_ds3_ratio}"

    def test_checkpointing(
        self,
        dataset_factory,
        small_dataset_file,
        medium_dataset_file,
        large_dataset_file,
    ):
        """Tests that interleaved dataset checkpointing preserves sampling state in nested structure."""

        def create_interleaved():
            ds1 = dataset_factory(small_dataset_file, dataset_name="ds1", weight=0.3)
            ds2 = dataset_factory(medium_dataset_file, dataset_name="ds2", weight=0.7)
            ds3 = dataset_factory(large_dataset_file, dataset_name="ds3", weight=1.0)

            # Create nested structure: interleaved([interleaved([ds1, ds2]), ds3])
            child_interleaved = InterleavedDataset(
                [ds1, ds2], seed=SEED, dataset_name="child"
            )
            return InterleavedDataset(
                [child_interleaved, ds3], seed=SEED, dataset_name="parent"
            )

        # Original run
        interleaved1 = create_interleaved()
        loader1 = StatefulDataLoader(
            interleaved1, batch_size=BATCH_SIZE, collate_fn=collate_with_metrics
        )
        aggregator1 = MetricsAggregator()

        # Resumed run
        interleaved2 = create_interleaved()
        loader2 = StatefulDataLoader(
            interleaved2, batch_size=BATCH_SIZE, collate_fn=collate_with_metrics
        )
        aggregator2 = MetricsAggregator()

        result = generate_ckpt(
            loader1,
            aggregator1,
            steps_before_checkpoint=10,
            steps_after_checkpoint=20,
            resume_dataloader=loader2,
            resume_aggregator=aggregator2,
        )

        orig_post_ids = [b["id"].tolist() for b in result["post_checkpoint_batches"]]
        resumed_ids = [b["id"].tolist() for b in result["resumed_batches"]]
        assert (
            orig_post_ids == resumed_ids
        ), "Resumed batches should be identical for deterministic run"
        assert (
            result["final_metrics"] == result["resumed_metrics"]
        ), "Final metrics should match"

        # Test sampling log functionality
        # Check that sampling log contains tuples of (iteration_count, dataset_name)
        state_dict = interleaved1.state_dict()
        sampling_log = state_dict["sampling_log"]
        iteration_count = state_dict["iteration_count"]

        assert len(sampling_log) > 0, "Sampling log should not be empty"
        assert iteration_count > 0, "Iteration count should be positive"

        # Check sampling ratios by analyzing the actual samples processed during the test
        # Since the sampling log only shows immediate children ("child", "ds3"),
        # we need to look at the actual sample IDs to determine leaf dataset usage

        # Collect all sample IDs from the batches processed during checkpointing
        all_sample_ids = []
        for batch_list in [
            result["pre_checkpoint_batches"],
            result["post_checkpoint_batches"],
        ]:
            for batch in batch_list:
                all_sample_ids.extend(batch["id"].tolist())

        # Count samples by ID ranges: ds1 has IDs 0-22, ds2 has IDs 100-134, ds3 has IDs 1000-1046
        ds1_count = sum(1 for id in all_sample_ids if 0 <= id < SMALL_DATASET_SIZE)
        ds2_count = sum(
            1 for id in all_sample_ids if 100 <= id < (MEDIUM_DATASET_SIZE + 100)
        )
        ds3_count = sum(
            1 for id in all_sample_ids if 1000 <= id < (LARGE_DATASET_SIZE + 1000)
        )
        total_samples = ds1_count + ds2_count + ds3_count
        ds1_ratio = ds1_count / total_samples
        ds2_ratio = ds2_count / total_samples
        ds3_ratio = ds3_count / total_samples

        # Expected ratios based on nested weighting:
        # Inner weights: ds1=0.3, ds2=0.7 -> inner total=1.0
        # Outer weights: inner=1.0, ds3=1.0 -> normalized to 0.5 each
        # Final ratios: ds1=0.5*0.3=0.15, ds2=0.5*0.7=0.35, ds3=0.5
        expected_ds1_ratio = 0.15
        expected_ds2_ratio = 0.35
        expected_ds3_ratio = 0.5

        # Allow larger tolerance due to small sample size in checkpointing test
        assert (
            abs(ds1_ratio - expected_ds1_ratio) < 0.2
        ), f"ds1 ratio {ds1_ratio:.2f} should be ~{expected_ds1_ratio}"
        assert (
            abs(ds2_ratio - expected_ds2_ratio) < 0.2
        ), f"ds2 ratio {ds2_ratio:.2f} should be ~{expected_ds2_ratio}"
        assert (
            abs(ds3_ratio - expected_ds3_ratio) < 0.2
        ), f"ds3 ratio {ds3_ratio:.2f} should be ~{expected_ds3_ratio}"


class TestDistributedInterleavedDataset(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @gpu_test(gpu_count=2)
    def test_distributed_interleaved_checkpointing(self):
        """
        Test interleaved dataset checkpointing with distributed settings using nested structure.
        Assertions:
        - Each rank processes non-overlapping data shards
        - Sampling ratios for nested structure (ds1: 15%, ds2: 35%, ds3: 50%) are maintained across ranks
        - Checkpoint/resume produces identical batches (deterministic)
        - Metrics correctly aggregate across ranks
        """
        rank = dist.get_rank()

        # Create shared temp directory (only rank 0 creates it)
        if rank == 0:
            temp_dir = tempfile.mkdtemp(prefix="interleaved_test_")
        else:
            temp_dir = None

        # Broadcast temp directory to all ranks
        temp_dir_list = [temp_dir] if temp_dir is not None else [""]
        dist.broadcast_object_list(temp_dir_list, src=0)
        temp_dir = temp_dir_list[0]
        tmp_path = Path(temp_dir)

        try:

            def create_dataset():
                file1 = tmp_path / "ds1.json"
                file2 = tmp_path / "ds2.json"
                file3 = tmp_path / "ds3.json"

                # Only rank 0 creates the data files
                if rank == 0:
                    create_test_json_file(file1, SMALL_DATASET_SIZE)  # IDs 0-22
                    create_test_json_file(
                        file2, MEDIUM_DATASET_SIZE, offset=100
                    )  # IDs 100-134
                    create_test_json_file(
                        file3, LARGE_DATASET_SIZE, offset=1000
                    )  # IDs 1000-1046
                dist.barrier()  # Wait for file creation

                ds1 = HfIterableDataset(
                    path="json",
                    data_files=str(file1),
                    split="train",
                    dataset_name="ds1",
                    shuffle_buffer_size=0,  # No shuffle for determinism
                    metric_transform=DefaultTrainingMetricTransform(),
                    num_shards_per_rank=2,
                    weight=0.3,
                )
                ds2 = HfIterableDataset(
                    path="json",
                    data_files=str(file2),
                    split="train",
                    dataset_name="ds2",
                    shuffle_buffer_size=0,  # No shuffle for determinism
                    metric_transform=DefaultTrainingMetricTransform(),
                    num_shards_per_rank=2,
                    weight=0.7,
                )
                ds3 = HfIterableDataset(
                    path="json",
                    data_files=str(file3),
                    split="train",
                    dataset_name="ds3",
                    shuffle_buffer_size=0,  # No shuffle for determinism
                    metric_transform=DefaultTrainingMetricTransform(),
                    num_shards_per_rank=2,
                    weight=1.0,
                )

                # Create nested structure: interleaved([interleaved([ds1, ds2]), ds3])
                child_interleaved = InterleavedDataset(
                    [ds1, ds2], seed=SEED, dataset_name="child"
                )
                return InterleavedDataset(
                    [child_interleaved, ds3], seed=SEED, dataset_name="parent"
                )

            def create_dataloader(dataset):
                loader = StatefulDataLoader(
                    dataset,
                    batch_size=BATCH_SIZE,
                    num_workers=0,  # Avoid multiprocessing in distributed tests
                    collate_fn=collate_with_metrics,
                )
                return loader, MetricsAggregator()

            # Run checkpointing test with small number of steps
            loader1, aggregator1 = create_dataloader(create_dataset())
            loader2, aggregator2 = create_dataloader(create_dataset())

            result = generate_ckpt(
                loader1,
                aggregator1,
                3,
                3,  # 3 steps before, 3 steps after checkpoint
                resume_dataloader=loader2,
                resume_aggregator=aggregator2,
            )

            # Verify deterministic resumption
            orig_post_ids = [
                b["id"].tolist() for b in result["post_checkpoint_batches"]
            ]
            resumed_ids = [b["id"].tolist() for b in result["resumed_batches"]]
            assert orig_post_ids == resumed_ids, (
                f"Rank {rank}: Non-deterministic interleaved resume. "
                f"This indicates sampling state is not properly preserved."
            )
            assert (
                result["final_metrics"] == result["resumed_metrics"]
            ), "Final metrics don't match resumed metrics - aggregator state issue"

            # Verify sampling ratio is approximately maintained for nested structure
            all_ids = []
            for batch in (
                result["pre_checkpoint_batches"] + result["post_checkpoint_batches"]
            ):
                all_ids.extend(batch["id"].tolist())

            # Count samples by ID ranges: ds1 has IDs 0-22, ds2 has IDs 100-134, ds3 has IDs 1000-1046
            ds1_samples = sum(1 for id in all_ids if 0 <= id < SMALL_DATASET_SIZE)
            ds2_samples = sum(
                1 for id in all_ids if 100 <= id < (MEDIUM_DATASET_SIZE + 100)
            )
            ds3_samples = sum(
                1 for id in all_ids if 1000 <= id < (LARGE_DATASET_SIZE + 1000)
            )
            total_samples = ds1_samples + ds2_samples + ds3_samples

            if total_samples > 0:
                ds1_ratio = ds1_samples / total_samples
                ds2_ratio = ds2_samples / total_samples
                ds3_ratio = ds3_samples / total_samples

                # Expected ratios based on nested weighting:
                # Inner weights: ds1=0.3, ds2=0.7 -> inner total=1.0
                # Outer weights: inner=1.0, ds3=1.0 -> normalized to 0.5 each
                # Final ratios: ds1=0.5*0.3=0.15, ds2=0.5*0.7=0.35, ds3=0.5
                expected_ds1_ratio = 0.15
                expected_ds2_ratio = 0.35
                expected_ds3_ratio = 0.5

                assert (
                    abs(ds1_ratio - expected_ds1_ratio) < 0.1
                ), f"ds1 ratio {ds1_ratio:.2f} should be ~{expected_ds1_ratio}"
                assert (
                    abs(ds2_ratio - expected_ds2_ratio) < 0.1
                ), f"ds2 ratio {ds2_ratio:.2f} should be ~{expected_ds2_ratio}"
                assert (
                    abs(ds3_ratio - expected_ds3_ratio) < 0.1
                ), f"ds3 ratio {ds3_ratio:.2f} should be ~{expected_ds3_ratio}"

        finally:
            # Clean up temp directory (only rank 0)
            if rank == 0:
                shutil.rmtree(temp_dir)
