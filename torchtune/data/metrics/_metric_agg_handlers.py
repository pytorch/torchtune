# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import torch

from torchtune.data.metrics._metric_transform import AggregationType, Metric

logger = logging.getLogger(__name__)


@dataclass
class MetricState:
    """Mutable state object representing aggregated metric for (dataset, metric) on a single rank.

    Attributes:
        dataset_name (str): Name of the dataset.
        metric_name (str): Name of the metric.
        value (float): Current aggregated value, whose meaning depends on the aggregation type
            (e.g., running sum, current max).
        agg_type (AggregationType): Aggregation type.
        metadata (dict[str, Any]): Additional state like count, list of values, etc.
    """

    dataset_name: str
    metric_name: str
    value: float
    agg_type: AggregationType
    metadata: dict[str, Any] = field(default_factory=dict)


class AggregationHandler(ABC):
    """Base class for handling metric aggregation in MetricsAggregator.

    This class defines the interface for different aggregation strategies (e.g., SUM, MEAN).
    Each handler is responsible for:
    - Initializing the state for a new (dataset, metric) pair.
    - Updating the state with new values.
    - Finalizing the value for local (single-rank) logging.
    - Reducing the values from all ranks in a distributed setting.
    - Serializing and deserializing the metric state for checkpointing.
    """

    @abstractmethod
    def initialize_metric_state(
        self, dataset_name: str, metric_name: str, agg_type: AggregationType
    ) -> MetricState:
        """Create a new MetricState for a (dataset_name, metric_name) pair.

        Args:
            dataset_name (str): Name of the dataset. Especially useful when tracking multiple datasets.
            metric_name (str): Name of the metric.
            agg_type (AggregationType): Aggregation type.

        Returns:
            MetricState: New MetricState for this (dataset_name, metric_name) pair.
        """
        pass

    @abstractmethod
    def update(self, local_agg_metric: MetricState, metric: Metric) -> None:
        """Update cumulative MetricState with new metric info.

        Args:
            local_agg_metric (MetricState): Cumulative state of the aggregation for this metric in the local rank.
            metric (Metric): Input metric info.
        """
        pass

    @abstractmethod
    def finalize_local_agg(self, local_agg_metric: MetricState) -> list[MetricState]:
        """
        Computes the final value from the locally aggregated state.

        This method may expand a single metric into multiple, for instance,
        a distribution into mean, min, max, and percentiles.

        Args:
            local_agg_metric (MetricState): The locally aggregated metric state to finalize.

        Returns:
            list[MetricState]: List of finalized metric states.
        """
        pass

    @abstractmethod
    def finalize_dist_agg(self, local_agg_metrics: list[MetricState]) -> MetricState:
        """
        Merge MetricStates from all ranks into final result.

        Args:
            local_agg_metrics (list[MetricState]): list of MetricStates for this (dataset_name, metric_name) pair.

        Returns:
            MetricState: Final result for this (dataset_name, metric_name) pair.
        """
        pass

    def serialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Convert handler-specific metadata to serializable format.

        Args:
            metadata (dict[str, Any]): AggHandler-specific metadata.

        Returns:
            dict[str, Any]: Serializable metadata.

        Override this when using non-serializable types like deque or Counter.
        For example, convert deque to list, Counter to dict.
        """
        return metadata.copy()

    def deserialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Restore handler-specific metadata from serialized format.

        Args:
            metadata (dict[str, Any]): AggHandler-specific metadata.

        Returns:
            dict[str, Any]: Deserialized metadata.

        Override this to reverse the serialize_metadata transformation.
        For example, convert list back to deque, dict back to Counter.
        """
        return metadata.copy()


class SumAggHandler(AggregationHandler):
    """AggHandler for SUM aggregation. Initializes with 0.0 and accumulates metric values."""

    def initialize_metric_state(
        self, dataset_name: str, metric_name: str, agg_type: AggregationType
    ) -> MetricState:
        return MetricState(
            dataset_name=dataset_name,
            metric_name=metric_name,
            value=0.0,
            agg_type=agg_type,
        )

    def update(self, local_agg_metric: MetricState, metric: Metric) -> None:
        if not isinstance(metric.value, (int, float)):
            raise ValueError(
                f"SumAggHandler expects numeric values, got {type(metric.value)}"
            )
        local_agg_metric.value += metric.value

    def finalize_local_agg(self, local_agg_metric: MetricState) -> list[MetricState]:
        return [local_agg_metric]

    def finalize_dist_agg(self, local_agg_metrics: list[MetricState]) -> MetricState:
        if not local_agg_metrics:
            raise ValueError("Cannot aggregate empty list of metrics")

        total = sum(metric.value for metric in local_agg_metrics)
        return MetricState(
            dataset_name=local_agg_metrics[0].dataset_name,
            metric_name=local_agg_metrics[0].metric_name,
            value=total,
            agg_type=local_agg_metrics[0].agg_type,
            metadata=local_agg_metrics[0].metadata.copy(),
        )


class MaxAggHandler(AggregationHandler):
    """AggHandler for MAX aggregation. Tracks maximum value across all updates."""

    def initialize_metric_state(
        self, dataset_name: str, metric_name: str, agg_type: AggregationType
    ) -> MetricState:
        return MetricState(
            dataset_name=dataset_name,
            metric_name=metric_name,
            value=float("-inf"),
            agg_type=agg_type,
        )

    def update(self, local_agg_metric: MetricState, metric: Metric) -> None:
        if not isinstance(metric.value, (int, float)):
            raise ValueError(
                f"MaxAggHandler expects numeric values, got {type(metric.value)}"
            )
        local_agg_metric.value = max(local_agg_metric.value, metric.value)

    def finalize_local_agg(self, local_agg_metric: MetricState) -> list[MetricState]:
        return [local_agg_metric]

    def finalize_dist_agg(self, local_agg_metrics: list[MetricState]) -> MetricState:
        max_value = max(r.value for r in local_agg_metrics)
        return MetricState(
            dataset_name=local_agg_metrics[0].dataset_name,
            metric_name=local_agg_metrics[0].metric_name,
            value=max_value,
            agg_type=local_agg_metrics[0].agg_type,
            metadata=local_agg_metrics[0].metadata.copy(),
        )


class MinAggHandler(AggregationHandler):
    """AggHandler for MIN aggregation. Tracks minimum value across all updates."""

    def initialize_metric_state(
        self, dataset_name: str, metric_name: str, agg_type: AggregationType
    ) -> MetricState:
        return MetricState(
            dataset_name=dataset_name,
            metric_name=metric_name,
            value=float("inf"),
            agg_type=agg_type,
        )

    def update(self, local_agg_metric: MetricState, metric: Metric) -> None:
        if not isinstance(metric.value, (int, float)):
            raise ValueError(
                f"MinAggHandler expects numeric values, got {type(metric.value)}"
            )
        local_agg_metric.value = min(local_agg_metric.value, metric.value)

    def finalize_local_agg(self, local_agg_metric: MetricState) -> list[MetricState]:
        return [local_agg_metric]

    def finalize_dist_agg(self, local_agg_metrics: list[MetricState]) -> MetricState:
        min_value = min(r.value for r in local_agg_metrics)
        return MetricState(
            dataset_name=local_agg_metrics[0].dataset_name,
            metric_name=local_agg_metrics[0].metric_name,
            value=min_value,
            agg_type=local_agg_metrics[0].agg_type,
            metadata=local_agg_metrics[0].metadata.copy(),
        )


class MeanAggHandler(AggregationHandler):
    """AggHandler for MEAN aggregation. Maintains running sum and count to compute average."""

    def initialize_metric_state(
        self, dataset_name: str, metric_name: str, agg_type: AggregationType
    ) -> MetricState:
        return MetricState(
            dataset_name=dataset_name,
            metric_name=metric_name,
            value=0.0,
            agg_type=agg_type,
            metadata={"sum": 0.0, "count": 0},
        )

    def update(self, local_agg_metric: MetricState, metric: Metric) -> None:
        local_agg_metric.metadata["sum"] += metric.value
        local_agg_metric.metadata["count"] += 1

    def finalize_local_agg(self, local_agg_metric: MetricState) -> list[MetricState]:
        count = local_agg_metric.metadata["count"]
        local_agg_metric.value = (
            local_agg_metric.metadata["sum"] / count if count > 0 else 0.0
        )
        return [local_agg_metric]

    def finalize_dist_agg(self, local_agg_metrics: list[MetricState]) -> MetricState:
        total_sum = sum(metric.metadata["sum"] for metric in local_agg_metrics)
        total_count = sum(metric.metadata["count"] for metric in local_agg_metrics)

        return MetricState(
            dataset_name=local_agg_metrics[0].dataset_name,
            metric_name=local_agg_metrics[0].metric_name,
            value=total_sum / total_count if total_count > 0 else 0.0,
            agg_type=local_agg_metrics[0].agg_type,
            metadata={"sum": total_sum, "count": total_count},
        )


class DistributionAggHandler(AggregationHandler):
    """AggHandler for DISTRIBUTION aggregation. Maintains a sliding window of values
    and expands into multiple statistical metrics (mean, min, max, percentiles, std).

    Note: Percentiles and standard deviation are approximated in distributed settings by averaging local
    percentiles and standard deviations across ranks. This is mathematically imprecise but provides a
    reasonable approximation for monitoring purposes.

    Args:
        window_size (int): Maximum number of recent values to retain for statistics.

    Raises:
            ValueError: If window_size is not positive.
    """

    def __init__(self, window_size: int = 1000):
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.window_size = window_size

    def initialize_metric_state(
        self, dataset_name: str, metric_name: str, agg_type: AggregationType
    ) -> MetricState:
        return MetricState(
            dataset_name=dataset_name,
            metric_name=metric_name,
            value=0.0,
            agg_type=agg_type,
            metadata={"values": deque(maxlen=self.window_size)},
        )

    def update(self, local_agg_metric: MetricState, metric: Metric) -> None:
        local_agg_metric.metadata["values"].append(metric.value)

    def finalize_local_agg(self, local_agg_metric: MetricState) -> list[MetricState]:
        values = list(local_agg_metric.metadata["values"])
        if not values:
            return []

        return self._compute_distribution_stats(local_agg_metric, values)

    def _compute_distribution_stats(
        self, local_agg_metric: MetricState, values: list[float]
    ) -> list[MetricState]:
        """Compute statistical metrics from distribution values using torch for efficiency."""
        if not values:
            return []

        # Use float64 for precision matching python's float
        values_tensor = torch.tensor(values, dtype=torch.float64)
        n = len(values_tensor)

        # Compute all stats from the tensor
        sum_val = torch.sum(values_tensor).item()
        mean_val = sum_val / n
        min_val = torch.min(values_tensor).item()
        max_val = torch.max(values_tensor).item()

        # Compute all percentiles in one go
        percentile_definitions = torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)
        p05_val, p50_val, p95_val = torch.quantile(
            values_tensor, percentile_definitions
        ).tolist()

        # Return multiple MetricStates with proper agg_types for distributed reduction
        # NOTE: Percentiles use MEAN aggregation which approximates global percentiles
        # by averaging local percentiles.
        metrics = [
            MetricState(
                dataset_name=local_agg_metric.dataset_name,
                metric_name=f"{local_agg_metric.metric_name}_stat_mean",
                value=mean_val,
                agg_type=AggregationType.MEAN,
                metadata={"sum": sum_val, "count": n},
            ),
            MetricState(
                dataset_name=local_agg_metric.dataset_name,
                metric_name=f"{local_agg_metric.metric_name}_stat_min",
                value=min_val,
                agg_type=AggregationType.MIN,
                metadata={},
            ),
            MetricState(
                dataset_name=local_agg_metric.dataset_name,
                metric_name=f"{local_agg_metric.metric_name}_stat_max",
                value=max_val,
                agg_type=AggregationType.MAX,
                metadata={},
            ),
            MetricState(
                dataset_name=local_agg_metric.dataset_name,
                metric_name=f"{local_agg_metric.metric_name}_stat_p05",
                value=p05_val,
                agg_type=AggregationType.MEAN,
                metadata={"sum": p05_val, "count": 1},
            ),
            MetricState(
                dataset_name=local_agg_metric.dataset_name,
                metric_name=f"{local_agg_metric.metric_name}_stat_p50",
                value=p50_val,
                agg_type=AggregationType.MEAN,
                metadata={"sum": p50_val, "count": 1},
            ),
            MetricState(
                dataset_name=local_agg_metric.dataset_name,
                metric_name=f"{local_agg_metric.metric_name}_stat_p95",
                value=p95_val,
                agg_type=AggregationType.MEAN,
                metadata={"sum": p95_val, "count": 1},
            ),
        ]
        # Standard deviation is only well-defined for n > 1
        if n > 1:
            std_val = torch.std(values_tensor).item()
            metrics.append(
                MetricState(
                    dataset_name=local_agg_metric.dataset_name,
                    metric_name=f"{local_agg_metric.metric_name}_stat_std",
                    value=std_val,
                    agg_type=AggregationType.MEAN,
                    metadata={"sum": std_val, "count": 1},
                )
            )
        return metrics

    def finalize_dist_agg(self, local_agg_metrics: list[MetricState]) -> MetricState:
        raise NotImplementedError(
            "Metrics with AggregationType.DISTRIBUTION are converted to other "
            "AggregationTypes for distributed reduction. finalize_dist_agg should not be called."
        )

    def serialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Convert deque to list for serialization."""
        serialized = metadata.copy()
        if "values" in serialized:
            serialized["values"] = list(serialized["values"])
        return serialized

    def deserialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Convert list back to deque."""
        deserialized = metadata.copy()
        if "values" in deserialized:
            deserialized["values"] = deque(
                deserialized["values"], maxlen=self.window_size
            )
        return deserialized


class CategoricalCountAggHandler(AggregationHandler):
    """AggHandler for CATEGORICAL_COUNT aggregation. Counts occurrences of categorical values
    and expands into individual count metrics for each category."""

    def initialize_metric_state(
        self, dataset_name: str, metric_name: str, agg_type: AggregationType
    ) -> MetricState:
        return MetricState(
            dataset_name=dataset_name,
            metric_name=metric_name,
            value=0.0,
            agg_type=agg_type,
            metadata={"counts": Counter()},
        )

    def update(self, local_agg_metric: MetricState, metric: Metric) -> None:
        local_agg_metric.metadata["counts"][metric.value] += 1

    def finalize_local_agg(self, local_agg_metric: MetricState) -> list[MetricState]:
        # Expand categorical counts into individual metrics
        results = []
        for category, count in local_agg_metric.metadata["counts"].items():
            results.append(
                MetricState(
                    dataset_name=local_agg_metric.dataset_name,
                    metric_name=f"{local_agg_metric.metric_name}_count_{category}",
                    value=count,
                    agg_type=AggregationType.SUM,
                )
            )
        return results

    def finalize_dist_agg(self, local_agg_metrics: list[MetricState]) -> MetricState:
        raise NotImplementedError(
            "Metrics with AggregationType.CATEGORICAL_COUNT are converted to other "
            "AggregationTypes for distributed reduction. finalize_dist_agg should not be called."
        )

    def serialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Convert Counter to dict for serialization."""
        serialized = metadata.copy()
        if "counts" in serialized:
            serialized["counts"] = dict(serialized["counts"])
        return serialized

    def deserialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Convert dict back to Counter."""
        deserialized = metadata.copy()
        if "counts" in deserialized:
            deserialized["counts"] = Counter(deserialized["counts"])
        return deserialized
