# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

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
