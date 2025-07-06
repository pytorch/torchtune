# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

from torchtune.modules.transforms import Transform


@dataclass(frozen=True)
class Metric:
    dataset_name: str
    name: str
    value: Union[int, float, str]
    agg_type: "AggregationType"


class AggregationType(Enum):
    """Defines how a metric's value should be aggregated."""

    SUM = "sum"
    MEAN = "mean"
    DISTRIBUTION = "distribution"
    CATEGORICAL_COUNT = "categorical_count"
    MAX = "max"
    MIN = "min"


class MetricTransform(Transform):
    """Applied to each dataset sample to generate per-sample metrics for training tracking.

    Creates Metric objects that are later aggregated by 'MetricsAggregator'. This separation
    of concerns ensures metrics are correctly aggregated even with multiple dataloader
    workers and in distributed settings."""

    def __init__(self):
        # dataset_name is set by the dataset using set_dataset_name
        self.dataset_name: Optional[str] = None

    def set_dataset_name(self, dataset_name: str) -> None:
        """Called by dataset to set the namespace for metrics.

        The dataset name is used to differentiate multiple datasets stats,
        e.g. "train/dataset1/tokens_seen" and "train/dataset2/tokens_seen".

        Args:
            dataset_name (str): Name of the dataset for metric namespacing
        """
        self.dataset_name = dataset_name

    def _generate_metrics(self, sample: dict[str, Any]) -> list[Metric]:
        """Generate metrics for a single sample. Must be implemented by subclasses.

        Args:
            sample (dict[str, Any]): The sample dictionary to generate metrics from

        Returns:
            list[Metric]: List of metrics generated for this sample

        Raises:
            NotImplementedError: If subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement _generate_metrics method")

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply transform to sample, adding generated metrics."""
        if self.dataset_name is None:
            raise RuntimeError(
                "set_dataset_name() must be called before using the transform."
            )

        # Generate metrics for this sample
        metrics = self._generate_metrics(sample)

        # Add to existing metrics list or create new one
        if "metrics" not in sample:
            sample["metrics"] = []
        sample["metrics"].extend(metrics)
        return sample


class DefaultTrainingMetricTransform(MetricTransform):
    """Generates training metrics: samples_seen, tokens_seen, seq_len distribution.

    For details about MetricTransform base class behavior, see the parent class docstring.

    Tracked metrics:
    - samples_seen: Cumulative count of samples processed (SUM aggregation)
    - tokens_seen: Cumulative sum of all tokens processed (SUM aggregation)
    - seq_len: Distribution of sequence lengths (DISTRIBUTION aggregation)

    Example:
        >>> transform = DefaultTrainingMetricTransform()
        >>> transform.set_dataset_name("alpaca")
        >>>
        >>> sample = {"tokens": [1, 2, 3, 4, 5]}  # 5 tokens
        >>> metrics = transform._generate_metrics(sample)
        >>> # Creates:
        >>> # [
        >>> #   Metric(dataset_name="alpaca", name="samples_seen", value=1, agg_type=AggregationType.SUM),
        >>> #   Metric(dataset_name="alpaca", name="tokens_seen", value=5, agg_type=AggregationType.SUM),
        >>> #   Metric(dataset_name="alpaca", name="seq_len", value=5, agg_type=AggregationType.DISTRIBUTION)
        >>> # ]
    """

    def _generate_metrics(self, sample: dict[str, Any]) -> list[Metric]:
        if self.dataset_name is None:
            raise RuntimeError(
                "set_dataset_name() must be called before using the transform."
            )

        # Determine token key
        token_key = "tokens" if "tokens" in sample else "input_ids"
        token_len = len(sample.get(token_key, []))

        # Create metrics for this sample
        return [
            Metric(
                dataset_name=self.dataset_name,
                name="samples_seen",
                value=1,
                agg_type=AggregationType.SUM,
            ),
            Metric(
                dataset_name=self.dataset_name,
                name="tokens_seen",
                value=token_len,
                agg_type=AggregationType.SUM,
            ),
            Metric(
                dataset_name=self.dataset_name,
                name="seq_len",
                value=token_len,
                agg_type=AggregationType.DISTRIBUTION,
            ),
        ]
