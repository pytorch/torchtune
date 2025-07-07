# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests cover:
- DefaultTrainingMetricTransform
- Basic metric generation (samples_seen, tokens_seen, seq_len)
- Dataset name validation and requirements
- Proper metric type assignment and aggregation configuration
"""

import pytest

from torchtune.data.metrics import AggregationType, DefaultTrainingMetricTransform


class TestDefaultTrainingMetricTransform:
    """Tests for DefaultTrainingMetricTransform functionality."""

    def test_dataset_name_not_set_raises_error(self):
        """Test that the transform raises a RuntimeError if used before
        `set_dataset_name` is called, ensuring that metrics are always
        correctly attributed to a dataset."""
        transform = DefaultTrainingMetricTransform()
        sample = {"tokens": [1, 2, 3]}

        with pytest.raises(RuntimeError, match="set_dataset_name"):
            transform(sample)

    def test_basic_metrics_generation(self):
        """Test that transform generates expected training metrics for input samples."""
        transform = DefaultTrainingMetricTransform()
        # Set dataset name required for metric generation
        transform.set_dataset_name("test_dataset")

        sample = {"tokens": [1, 2, 3, 4, 5]}
        result = transform(sample)

        # Transform should preserve original sample data unchanged
        assert result["tokens"] == [1, 2, 3, 4, 5]

        # Should generate exactly 3 metrics: samples_seen, tokens_seen, seq_len
        assert "metrics" in result
        metrics = result["metrics"]
        assert len(metrics) == 3

        # Verify each metric has correct properties and aggregation type
        for metric in metrics:
            if metric.metric_name == "samples_seen":
                assert metric.dataset_name == "test_dataset"
                assert metric.value == 1
                assert metric.agg_type == AggregationType.SUM

            elif metric.metric_name == "tokens_seen":
                assert metric.dataset_name == "test_dataset"
                assert metric.value == 5
                assert metric.agg_type == AggregationType.SUM

            elif metric.metric_name == "seq_len":
                assert metric.dataset_name == "test_dataset"
                assert metric.value == 5
                assert metric.agg_type == AggregationType.DISTRIBUTION
