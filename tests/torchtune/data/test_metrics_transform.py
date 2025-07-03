# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtune.data.metrics import AggregationType, DefaultTrainingMetricTransform


class TestDefaultTrainingMetricTransform:
    """Tests for DefaultTrainingMetricTransform functionality."""

    def test_dataset_name_not_set_raises_error(self):
        """Test that using transform without setting dataset name raises error."""
        transform = DefaultTrainingMetricTransform()
        sample = {"tokens": [1, 2, 3]}

        with pytest.raises(RuntimeError, match="set_dataset_name"):
            transform(sample)

    def test_basic_metrics_generation(self):
        """Test that transform generates expected metrics for a sample."""
        transform = DefaultTrainingMetricTransform()
        transform.set_dataset_name("test_dataset")

        sample = {"tokens": [1, 2, 3, 4, 5]}
        result = transform(sample)

        # Should preserve original sample data
        assert result["tokens"] == [1, 2, 3, 4, 5]

        # Should add metrics
        assert "metrics" in result
        metrics = result["metrics"]
        assert len(metrics) == 3

        # Check each metric
        for metric in metrics:
            if metric.name == "samples_seen":
                assert metric.dataset_name == "test_dataset"
                assert metric.value == 1
                assert metric.agg_type == AggregationType.SUM

            elif metric.name == "tokens_seen":
                assert metric.dataset_name == "test_dataset"
                assert metric.value == 5
                assert metric.agg_type == AggregationType.SUM

            elif metric.name == "seq_len":
                assert metric.dataset_name == "test_dataset"
                assert metric.value == 5
                assert metric.agg_type == AggregationType.DISTRIBUTION
