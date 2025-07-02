# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
import collections
import logging
from typing import Any

import torch.distributed as dist

from torchtune.data.metrics import AggregationType, Metric

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """
    Aggregates metrics across datasets and distributed ranks.

    The internal state `_state` is a dictionary where the key is a tuple
    of `(dataset_name, metric_name)` and the value is another dictionary
    holding the metric's specific state (e.g., `{'type': AggregationType.SUM, 'value': 10}`).

    Usage:
        aggregator = MetricsAggregator()
        aggregator.update(metrics)
        # Get logger-ready metrics {key: value}
        metrics = aggregator.get_metrics_for_logging(prefix="train")  # {"train/dataset1/tokens": 1234, ...}
    """

    def __init__(self, dist_window_size: int = 1000):
        # State shape: {(dataset_name, metric_name): {type: AggType, value/sum/counts/etc}}
        self._state: dict[tuple[str, str], dict[str, Any]] = {}

        # For distributions, we keep a window of values to compute percentiles
        self._dist_window_size = dist_window_size

    def update(self, metrics: list[Metric]) -> None:
        """Update internal state with new metrics.

        Args:
            metrics (list[Metric]): list of Metric objects
        """
        for metric in metrics:
            key = (metric.dataset_name, metric.name)

            if key not in self._state:
                self._initialize_state(key, metric.agg_type)

            state = self._state[key]

            # Update based on aggregation type
            if metric.agg_type == AggregationType.SUM:
                state["value"] += metric.value
            elif metric.agg_type == AggregationType.MAX:
                if state["value"] is not None:
                    state["value"] = max(state["value"], metric.value)
                else:
                    state["value"] = metric.value
            elif metric.agg_type == AggregationType.MIN:
                if state["value"] is not None:
                    state["value"] = min(state["value"], metric.value)
                else:
                    state["value"] = metric.value
            elif metric.agg_type == AggregationType.MEAN:
                state["sum"] += metric.value
                state["count"] += 1
            elif metric.agg_type == AggregationType.DISTRIBUTION:
                state["values"].append(metric.value)
            elif metric.agg_type == AggregationType.CATEGORICAL_COUNT:
                state["counts"][metric.value] += 1

    def _initialize_state(
        self, key: tuple[str, str], agg_type: AggregationType
    ) -> None:
        """Initialize state for a new metric."""
        self._state[key] = {"type": agg_type}
        state = self._state[key]

        if agg_type == AggregationType.SUM:
            state["value"] = 0.0
        elif agg_type in (AggregationType.MAX, AggregationType.MIN):
            state["value"] = None
        elif agg_type == AggregationType.MEAN:
            state["sum"] = 0.0
            state["count"] = 0
        elif agg_type == AggregationType.DISTRIBUTION:
            state["values"] = collections.deque(maxlen=self._dist_window_size)
        elif agg_type == AggregationType.CATEGORICAL_COUNT:
            state["counts"] = collections.Counter()

    def get_metrics_for_logging(self, prefix: str = "data") -> dict[str, float]:
        """
        Returns aggregated metrics ready for logging to wandb/tensorboard.

        Args:
            prefix (str): Optional prefix like "train" or "valid" for metric keys

        Returns:
            dict[str, float]: Flat dictionary with keys like "train/dataset1/tokens_seen" -> float value
            Ready to be logged directly: wandb.log(metrics)
        """
        # Always compute local metrics first
        local_metrics = self._compute_local_metrics()

        # In distributed mode, perform reduction
        if dist.is_initialized() and dist.get_world_size() > 1:
            metrics = self._compute_distributed_metrics(local_metrics)
        else:
            metrics = local_metrics

        # Format for logging with proper key structure
        return self._format_for_logging(metrics, prefix)

    def _compute_local_metrics(self) -> dict[tuple[str, str], dict[str, Any]]:
        """
        Compute metrics from current state.

        For distributions and categoricals, expands into multiple entries.
        The dict format allows future extensions with additional fields.

        Returns:
            dict[tuple[str, str], dict[str, Any]]: dictionary mapping
                (dataset_name, metric_name) -> {"value": value, "agg_type": aggregation_type}
        """
        metrics = {}

        for (ds_name, metric_name), state in self._state.items():
            agg_type = state["type"]

            if agg_type in (
                AggregationType.SUM,
                AggregationType.MAX,
                AggregationType.MIN,
            ):
                # For sum, max, and min, we just need to return the value
                metrics[(ds_name, metric_name)] = {
                    "value": state["value"],
                    "agg_type": agg_type,
                }

            elif agg_type == AggregationType.MEAN:
                if state["count"] > 0:
                    value = state["sum"] / state["count"]
                    metrics[(ds_name, metric_name)] = {
                        "value": value,
                        "agg_type": agg_type,
                    }

            elif agg_type == AggregationType.DISTRIBUTION:
                # queue -> list
                values = list(state["values"])

                # Sort to get percentiles efficiently
                sorted_values = sorted(values)
                n = len(sorted_values)

                # Each stat becomes its own metric
                # so that we can all gather O(5) values across ranks
                # instead of the entire distribution
                metrics[(ds_name, f"{metric_name}_mean")] = {
                    "value": sum(values) / n,
                    "agg_type": AggregationType.MEAN,
                }
                metrics[(ds_name, f"{metric_name}_min")] = {
                    "value": sorted_values[0],
                    "agg_type": AggregationType.MIN,
                }
                metrics[(ds_name, f"{metric_name}_max")] = {
                    "value": sorted_values[-1],
                    "agg_type": AggregationType.MAX,
                }
                metrics[(ds_name, f"{metric_name}_p05")] = {
                    "value": sorted_values[max(0, int(0.05 * n) - 1)],
                    "agg_type": AggregationType.MEAN,
                }
                metrics[(ds_name, f"{metric_name}_p50")] = {
                    "value": sorted_values[max(0, int(0.5 * n) - 1)],
                    "agg_type": AggregationType.MEAN,
                }
                metrics[(ds_name, f"{metric_name}_p95")] = {
                    "value": sorted_values[max(0, int(0.95 * n) - 1)],
                    "agg_type": AggregationType.MEAN,
                }

            elif agg_type == AggregationType.CATEGORICAL_COUNT:
                # Expand categorical counts into individual metrics
                for category, count in state["counts"].items():
                    metrics[(ds_name, f"{metric_name}_{category}_count")] = {
                        "value": count,
                        "agg_type": AggregationType.SUM,
                    }

        return metrics

    def _compute_distributed_metrics(
        self, local_metrics: dict[tuple[str, str], dict[str, Any]]
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """
        Performs distributed reduction on metrics.

        Strategy:
        1. Do a single all_gather_object to collect all metrics from all ranks
        2. Group metrics by key and aggregation type
        3. Apply the appropriate reduction operation locally

        This avoids complex tensor operations and handles all reduction in one pass.

        Args:
            local_metrics (dict[tuple[str, str], dict[str, Any]]): dict mapping
                (dataset, metric) -> {"value": value, "agg_type": agg_type, ...}

        Returns:
            dict[tuple[str, str], dict[str, Any]]: Reduced metrics in same format as input

        Example:
            rank_1_metrics =
            {
                ("ds1", "metric1"): {"value": 10, "agg_type": AggregationType.SUM},
                ("ds2", "metric2"): {"value": 20, "agg_type": AggregationType.MEAN},
            }
            rank_2_metrics =
            {
                ("ds1", "metric1"): {"value": 30, "agg_type": AggregationType.SUM},
                ("ds2", "metric2"): {"value": 40, "agg_type": AggregationType.MEAN},
            }

            # After reduction
            result =
            {
                ("ds1", "metric1"): {"value": 40, "agg_type": AggregationType.SUM},
                ("ds2", "metric2"): {"value": 30, "agg_type": AggregationType.MEAN},
            }
        """
        world_size = dist.get_world_size()

        # Gather all metrics from all ranks in one operation
        all_metrics = [None] * world_size
        dist.all_gather_object(all_metrics, local_metrics)

        # Group values by key for reduction
        grouped = collections.defaultdict(list)
        for rank_metrics in all_metrics:
            if rank_metrics:  # It's possible a rank has no metrics
                for key, metric_dict in rank_metrics.items():
                    # A key is a tuple (dataset, metric)
                    grouped[key].append(metric_dict)

        # Reduce based on aggregation type
        reduced = {}
        if not grouped:
            return reduced

        for key, metric_dicts in grouped.items():
            # All metrics for a key should have same type, just take first
            values = [m["value"] for m in metric_dicts]
            agg_type = metric_dicts[0]["agg_type"]

            # Start with copy of first dict to preserve any extra fields
            result_dict = metric_dicts[0].copy()

            if agg_type == AggregationType.SUM:
                result_dict["value"] = sum(values)
            elif agg_type == AggregationType.MAX:
                result_dict["value"] = max(values)
            elif agg_type == AggregationType.MIN:
                result_dict["value"] = min(values)
            elif agg_type == AggregationType.MEAN:
                result_dict["value"] = sum(values) / len(values)

            reduced[key] = result_dict

        return reduced

    def _format_for_logging(
        self,
        metrics: dict[tuple[str, str], dict[str, Any]],
        prefix: str,
        template: str = r"{prefix}_{ds_name}/{metric_name}",
    ) -> dict[str, float]:
        """
        Format metrics for wandb/tensorboard logging.

        Args:
            metrics (dict[tuple[str, str], dict[str, Any]]): dict mapping
                (dataset, metric) -> {"value": value, "agg_type": agg_type, ...}
            prefix (str): Optional prefix like "train" or "valid"
            template (str): Template for metric key. Use {prefix}, {ds_name}, and {metric_name} as placeholders.

        Returns:
            dict[str, float]: Flat dict with string keys like "train/dataset1/tokens_seen" -> float
        """
        formatted = {}

        for (ds_name, metric_name), metric_dict in metrics.items():
            # Use regex format to build key
            key = template.format(
                prefix=prefix, ds_name=ds_name, metric_name=metric_name
            )
            formatted[key] = metric_dict["value"]

        return formatted

    def state_dict(self) -> dict[str, Any]:
        """Serialize aggregator state. The state is almost directly serializable."""
        serializable_state = {}
        for key, state in self._state.items():
            state_copy = state.copy()

            # Convert non-serializable types
            if "values" in state_copy:
                state_copy["values"] = list(state_copy["values"])  # deque → list
            if "counts" in state_copy:
                state_copy["counts"] = dict(state_copy["counts"])  # Counter → dict

            # Convert tuple key to string for JSON compatibility
            # JSON doesn't support tuple keys, so we convert (dataset, metric) → "('dataset', 'metric')"
            serializable_state[str(key)] = state_copy
        return {"state": serializable_state, "dist_window_size": self._dist_window_size}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load aggregator state from checkpoint."""
        self._dist_window_size = state_dict["dist_window_size"]

        deserialized_state = {}
        for key_str, state in state_dict["state"].items():
            # Convert string keys back to tuples
            # "('dataset', 'metric')" → ('dataset', 'metric')
            key = ast.literal_eval(key_str)

            # Re-wrap values in their original types
            if state.get("type") == AggregationType.DISTRIBUTION:
                state["values"] = collections.deque(
                    state["values"], maxlen=self._dist_window_size
                )
            if state.get("type") == AggregationType.CATEGORICAL_COUNT:
                state["counts"] = collections.Counter(state["counts"])

            deserialized_state[key] = state
        self._state = deserialized_state
