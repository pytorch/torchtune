# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch.distributed as dist
from torch.testing._internal.common_fsdp import FSDPTest
from tests.test_utils import gpu_test

from torchtune.data.metrics import AggregationType, Metric, MetricsAggregator


class TestMetricsAggregator:
    """Focused tests for MetricsAggregator functionality."""

    @pytest.mark.parametrize(
        "agg_type,test_values,expected",
        [
            (AggregationType.SUM, [1, 2, 3, 4], 10),
            (AggregationType.MEAN, [10, 20, 30, 40], 25.0),
            (AggregationType.MAX, [-5, 10, 3, 15], 15),
            (AggregationType.MIN, [5, -2, 8, 1], -2),
            (
                AggregationType.CATEGORICAL_COUNT,
                ["A", "B", "A", "C", "A"],
                {"A": 3, "B": 1, "C": 1},
            ),
        ],
    )
    def test_aggregation_types(self, agg_type, test_values, expected):
        """Tests each `AggregationType` to ensure it computes the correct value."""
        aggregator = MetricsAggregator()

        metrics = [
            Metric(dataset_name="test", name="metric", value=val, agg_type=agg_type)
            for val in test_values
        ]
        aggregator.update(metrics)

        result = aggregator.get_metrics_for_logging(prefix="train")

        if agg_type == AggregationType.CATEGORICAL_COUNT:
            for category, count in expected.items():
                assert result[f"train_test/metric_{category}_count"] == count
        else:
            assert result["train_test/metric"] == expected

    def test_distribution_metrics(self):
        """Tests that `AggregationType.DISTRIBUTION` computes all expected statistics (mean, min, max, p50)."""
        aggregator = MetricsAggregator()
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        metrics = [
            Metric("test", "dist_metric", val, AggregationType.DISTRIBUTION)
            for val in values
        ]
        aggregator.update(metrics)

        result = aggregator.get_metrics_for_logging(prefix="train")

        # Verify distribution statistics
        assert result["train_test/dist_metric_mean"] == 5.5
        assert result["train_test/dist_metric_min"] == 1
        assert result["train_test/dist_metric_max"] == 10
        assert (
            result["train_test/dist_metric_p50"] == 5
        )  # Median of 1-10 is 5 (index 4, value 5)

    def test_state_management(self):
        """Test aggregator checkpointing and restoration."""
        # Create aggregator with some state
        aggregator1 = MetricsAggregator()
        initial_metrics = [
            Metric("ds1", "counter", 10, AggregationType.SUM),
            Metric("ds1", "average", 5.0, AggregationType.MEAN),
            Metric("ds2", "categories", "X", AggregationType.CATEGORICAL_COUNT),
        ]
        aggregator1.update(initial_metrics)

        # Save state
        state = aggregator1.state_dict()

        # Create new aggregator and restore state
        aggregator2 = MetricsAggregator()
        aggregator2.load_state_dict(state)

        # Both should have identical metrics
        metrics1 = aggregator1.get_metrics_for_logging(prefix="train")
        metrics2 = aggregator2.get_metrics_for_logging(prefix="train")
        assert metrics1 == metrics2

        # Continue updating both - should remain identical
        additional_metrics = [
            Metric("ds1", "counter", 5, AggregationType.SUM),
            Metric("ds1", "average", 15.0, AggregationType.MEAN),
        ]
        aggregator1.update(additional_metrics)
        aggregator2.update(additional_metrics)

        final_metrics1 = aggregator1.get_metrics_for_logging(prefix="train")
        final_metrics2 = aggregator2.get_metrics_for_logging(prefix="train")
        assert final_metrics1 == final_metrics2

        # Verify expected values
        assert final_metrics1["train_ds1/counter"] == 15  # 10 + 5
        assert final_metrics1["train_ds1/average"] == 10.0  # (5 + 15) / 2

    def test_multiple_datasets(self):
        """Test that metrics from multiple datasets are correctly namespaced."""
        aggregator = MetricsAggregator()

        metrics = [
            Metric("dataset1", "samples", 100, AggregationType.SUM),
            Metric("dataset2", "samples", 200, AggregationType.SUM),
            Metric("dataset1", "tokens", 1000, AggregationType.SUM),
            Metric("dataset2", "tokens", 2000, AggregationType.SUM),
        ]
        aggregator.update(metrics)

        result = aggregator.get_metrics_for_logging(prefix="train")

        assert result["train_dataset1/samples"] == 100
        assert result["train_dataset2/samples"] == 200
        assert result["train_dataset1/tokens"] == 1000
        assert result["train_dataset2/tokens"] == 2000

    def test_empty_aggregator(self):
        """Test that empty aggregator returns empty metrics."""
        aggregator = MetricsAggregator()
        result = aggregator.get_metrics_for_logging(prefix="train")
        assert result == {}

    def test_prefix_handling(self):
        """Test that prefix is correctly applied to metric keys."""
        aggregator = MetricsAggregator()
        metrics = [
            Metric("test_ds", "metric1", 42, AggregationType.SUM),
            Metric("test_ds", "metric2", 84, AggregationType.SUM),
        ]
        aggregator.update(metrics)

        # Test with prefix
        result_with_prefix = aggregator.get_metrics_for_logging(prefix="validation")
        assert result_with_prefix["validation_test_ds/metric1"] == 42
        assert result_with_prefix["validation_test_ds/metric2"] == 84

        # Test without prefix (uses default "data")
        result_no_prefix = aggregator.get_metrics_for_logging()
        assert result_no_prefix["data_test_ds/metric1"] == 42
        assert result_no_prefix["data_test_ds/metric2"] == 84


class TestDistributedMetricsAggregator(FSDPTest):
    """Distributed tests for MetricsAggregator using FSDPTest infrastructure."""

    @property
    def world_size(self) -> int:
        return 2

    @gpu_test(gpu_count=2)
    def test_distributed_all_aggregation_types(self):
        """
        Test that all aggregation types work correctly in distributed setting.
        Each rank contributes different values to ensure proper reduction across ranks.
        """
        aggregator = MetricsAggregator()
        rank = dist.get_rank()

        # Each rank contributes different values to test cross-rank aggregation
        base_value = (rank + 1) * 10  # rank 0: 10, rank 1: 20

        metrics = [
            Metric("test", "sum_metric", base_value, AggregationType.SUM),
            Metric("test", "mean_metric", base_value + 5, AggregationType.MEAN),
            Metric("test", "max_metric", base_value * 10, AggregationType.MAX),
            Metric("test", "min_metric", base_value // 2, AggregationType.MIN),
        ]

        # DISTRIBUTION: Each rank adds 5 values for distribution statistics
        # rank 0: [0, 1, 2, 3, 4], rank 1: [10, 11, 12, 13, 14]
        for i in range(5):
            metrics.append(
                Metric("test", "dist_metric", rank * 10 + i, AggregationType.DISTRIBUTION)
            )

        # CATEGORICAL_COUNT: Different categories per rank to test counting
        # rank 0: 3 of cat_A, 2 of cat_B
        # rank 1: 1 of cat_A, 4 of cat_C
        if rank == 0:
            metrics.extend([
                Metric("test", "cat_metric", "cat_A", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "cat_A", AggregationType.CATEGORICAL_COUNT), 
                Metric("test", "cat_metric", "cat_A", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "cat_B", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "cat_B", AggregationType.CATEGORICAL_COUNT),
            ])
        else:
            metrics.extend([
                Metric("test", "cat_metric", "cat_A", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "cat_C", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "cat_C", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "cat_C", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "cat_C", AggregationType.CATEGORICAL_COUNT),
            ])

        # Update aggregator and get results
        aggregator.update(metrics)
        result = aggregator.get_metrics_for_logging(prefix="train")

        # Verify aggregation results across all ranks
        # SUM: rank 0 adds 10, rank 1 adds 20 -> total 30
        # MEAN: rank 0 has 15, rank 1 has 25 -> avg 20  
        # MAX: rank 0 has 100, rank 1 has 200 -> max 200
        # MIN: rank 0 has 5, rank 1 has 10 -> min 5
        assert result["train_test/sum_metric"] == 30
        assert result["train_test/mean_metric"] == 20
        assert result["train_test/max_metric"] == 200
        assert result["train_test/min_metric"] == 5

        # DISTRIBUTION: Combined values [0,1,2,3,4,10,11,12,13,14]
        # Mean should be average of local means: (2 + 12) / 2 = 7
        assert result["train_test/dist_metric_mean"] == 7
        assert result["train_test/dist_metric_min"] == 0
        assert result["train_test/dist_metric_max"] == 14

        # CATEGORICAL_COUNT: Total counts across ranks
        # cat_A: 3(rank0) + 1(rank1) = 4, cat_B: 2(rank0) + 0(rank1) = 2, cat_C: 0(rank0) + 4(rank1) = 4
        assert result["train_test/cat_metric_cat_A_count"] == 4
        assert result["train_test/cat_metric_cat_B_count"] == 2
        assert result["train_test/cat_metric_cat_C_count"] == 4

    @gpu_test(gpu_count=2)
    def test_distributed_state_dict_resumption(self):
        """
        Test that MetricsAggregator state_dict save/restore works correctly in distributed setting.
        Verifies:
        - State can be saved after partial updates across ranks
        - State can be restored consistently across ranks  
        - Continued updates after restore produce identical results
        - Distributed aggregation works correctly after restoration
        """
        rank = dist.get_rank()

        # Phase 1: Create aggregator and add initial metrics
        aggregator1 = MetricsAggregator()
        
        # Each rank contributes different initial values
        base_value = rank * 100  # rank 0: 0, rank 1: 100
        
        initial_metrics = [
            Metric("test", "sum_metric", base_value, AggregationType.SUM),
            Metric("test", "mean_metric", base_value // 2, AggregationType.MEAN),
            Metric("test", "max_metric", base_value * 2, AggregationType.MAX),
        ]
        
        # Add some DISTRIBUTION values - each rank adds 3 values
        for i in range(3):
            initial_metrics.append(
                Metric("test", "dist_metric", rank * 100 + i, AggregationType.DISTRIBUTION)
            )
        
        # Add CATEGORICAL_COUNT values
        if rank == 0:
            initial_metrics.extend([
                Metric("test", "cat_metric", "type_A", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "type_A", AggregationType.CATEGORICAL_COUNT),
            ])
        else:
            initial_metrics.extend([
                Metric("test", "cat_metric", "type_B", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "type_B", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "type_B", AggregationType.CATEGORICAL_COUNT),
            ])
        
        aggregator1.update(initial_metrics)
        
        # Save state_dict after initial update
        state_dict = aggregator1.state_dict()
        
        # Phase 2: Create new aggregator and restore from state_dict
        aggregator2 = MetricsAggregator()
        aggregator2.load_state_dict(state_dict)
        
        # Verify both aggregators produce identical results after restore
        result1 = aggregator1.get_metrics_for_logging(prefix="checkpoint")
        result2 = aggregator2.get_metrics_for_logging(prefix="checkpoint")
        assert result1 == result2, (
            f"Rank {rank}: Aggregators differ after state_dict restore"
        )
        
        # Phase 3: Add more metrics to both aggregators        
        additional_metrics = [
            Metric("test", "sum_metric", rank * 1000, AggregationType.SUM),
            Metric("test", "min_metric", rank * 1000, AggregationType.MIN),
        ]
        
        # Update both aggregators with additional metrics
        aggregator1.update(additional_metrics)
        aggregator2.update(additional_metrics)
        
        # Phase 4: Verify final results are identical across both aggregators
        final_result1 = aggregator1.get_metrics_for_logging(prefix="final")
        final_result2 = aggregator2.get_metrics_for_logging(prefix="final")
        assert final_result1 == final_result2, (
            f"Rank {rank}: Final results differ after continued updates"
        )
