# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import islice
from pathlib import Path
from unittest.mock import patch

import pytest

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtune.data import MetricsAggregator, StandardMetricTransform
from torchtune.datasets import HfIterableDataset, InterleavedDataset

# Import test utilities
from .test_iterable_utils import collate_with_metrics, generate_ckpt

# Test Constants
SMALL_DATASET_SIZE = 23
MEDIUM_DATASET_SIZE = 35
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


class TestInterleavedDataset:
    """Tests for multi-dataset interleaving functionality."""

    def test_initialization_validation(self, dataset_factory, small_dataset_file):
        """Tests that the dataset raises errors for invalid configurations, like duplicate names."""
        # Test duplicate dataset names
        ds1 = dataset_factory(small_dataset_file, dataset_name="duplicate")
        ds2 = dataset_factory(small_dataset_file, dataset_name="duplicate")

        with pytest.raises(ValueError, match="Duplicate dataset names detected"):
            InterleavedDataset(datasets=[ds1, ds2], weights=[0.5, 0.5], seed=SEED)

        # Test weight normalization (should work with warning)
        ds3 = dataset_factory(small_dataset_file, dataset_name="ds3")
        ds4 = dataset_factory(small_dataset_file, dataset_name="ds4")

        with patch("logging.Logger.warning") as mock_warning:
            interleaved = InterleavedDataset(
                datasets=[ds3, ds4],
                weights=[0.5, 1.5],
                seed=SEED,
                dataset_name="test_interleaved",  # Sum = 2.0 != 1.0
            )

            # Check that weights were normalized
            assert torch.allclose(interleaved._weights, torch.tensor([0.25, 0.75]))
            mock_warning.assert_called_once()

            assert interleaved.dataset_name == "test_interleaved"

    def test_sampling_ratios(
        self, dataset_factory, small_dataset_file, medium_dataset_file
    ):
        """Tests that datasets are sampled according to their assigned weights."""
        # Create two datasets with distinct ID ranges
        # ds1 has IDs 0-22 (small dataset)
        # ds2 has IDs 100-134 (medium dataset with offset)
        ds1 = dataset_factory(small_dataset_file, dataset_name="ds1")
        ds2 = dataset_factory(medium_dataset_file, dataset_name="ds2")

        # Test with 70/30 weighting
        weights = [0.7, 0.3]
        interleaved = InterleavedDataset([ds1, ds2], weights, seed=SEED)

        # Collect 300 samples
        sample_count = 300
        samples = list(islice(iter(interleaved), sample_count))

        # Count samples by checking ID ranges
        # ds1 has IDs < 100, ds2 has IDs >= 100
        ds1_count = sum(1 for s in samples if s["id"] < 100)
        ds2_count = sum(1 for s in samples if s["id"] >= 100)

        assert ds1_count + ds2_count == sample_count

        # Check ratios are approximately correct
        ds1_ratio = ds1_count / sample_count
        ds2_ratio = ds2_count / sample_count

        # Allow 10% tolerance due to randomness
        assert abs(ds1_ratio - 0.7) < 0.1, f"ds1 ratio {ds1_ratio:.2f} should be ~0.7"
        assert abs(ds2_ratio - 0.3) < 0.1, f"ds2 ratio {ds2_ratio:.2f} should be ~0.3"

    def test_metrics_aggregation(
        self, dataset_factory, small_dataset_file, medium_dataset_file
    ):
        """Tests that metrics from all child datasets are collected and aggregated."""
        ds1 = dataset_factory(small_dataset_file, dataset_name="ds1")
        ds2 = dataset_factory(medium_dataset_file, dataset_name="ds2")

        interleaved = InterleavedDataset([ds1, ds2], [0.2, 0.8], seed=SEED)
        aggregator = MetricsAggregator()

        # Process some samples
        total_samples = 200
        for sample in islice(iter(interleaved), total_samples):
            aggregator.update(sample["metrics"])

        metrics = aggregator.get_metrics_for_logging()

        # Should have metrics from both datasets, with flat keys
        assert "ds1/samples_seen" in metrics
        assert "ds2/samples_seen" in metrics

        # Both datasets should have contributed samples
        assert metrics["ds1/samples_seen"] > 0
        assert metrics["ds2/samples_seen"] > 0

        # Total samples should equal what we processed
        calculated_total_samples = (
            metrics["ds1/samples_seen"] + metrics["ds2/samples_seen"]
        )
        assert calculated_total_samples == total_samples

        # Test that ratio is approximately correct
        ds1_ratio = metrics["ds1/samples_seen"] / total_samples
        ds2_ratio = metrics["ds2/samples_seen"] / total_samples

        # Allow 10% tolerance due to randomness
        assert abs(ds1_ratio - 0.2) < 0.1, f"ds1 ratio {ds1_ratio:.2f} should be ~0.2"
        assert abs(ds2_ratio - 0.8) < 0.1, f"ds2 ratio {ds2_ratio:.2f} should be ~0.8"

    def test_checkpointing(
        self, dataset_factory, small_dataset_file, medium_dataset_file
    ):
        """Tests that interleaved dataset checkpointing preserves sampling state."""

        def create_interleaved():
            ds1 = dataset_factory(small_dataset_file, dataset_name="ds1")
            ds2 = dataset_factory(medium_dataset_file, dataset_name="ds2")
            return InterleavedDataset([ds1, ds2], [0.7, 0.3], seed=SEED)

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
