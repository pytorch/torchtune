# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
import logging
from collections import defaultdict
from typing import Any, Union

import torch.distributed as dist

from torchtune.data.metrics._metric_agg_handlers import (
    AggregationHandler,
    CategoricalCountAggHandler,
    DistributionAggHandler,
    MaxAggHandler,
    MeanAggHandler,
    MetricState,
    MinAggHandler,
    SumAggHandler,
)
from torchtune.data.metrics._metric_transform import AggregationType, Metric

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """Aggregates metrics across datasets and distributed ranks using pluggable handlers.

    Uses a handler-based strategy pattern where each aggregation type (SUM, MEAN, etc.)
    has its own handler. Maintains only one state per (dataset, metric) pair.

    When preparing for logging, uses a two-phase approach:
    1. Local aggregation: Each rank aggregates its metrics independently
    2. Distributed reduction: Results combined across ranks

    The aggregator is checkpointable and restores from state_dict for training resumption.

    Args:
        dist_window_size (int): Window size for DistributionAggHandler tracking.

    Example:
        >>> from torchtune.data.metrics import MetricsAggregator, Metric, AggregationType
        >>>
        >>> # Create aggregator
        >>> aggregator = MetricsAggregator()
        >>>
        >>> # Sample metrics from different batches
        >>> batch1_metrics = [
        ...     Metric("alpaca", "tokens_seen", 100, AggregationType.SUM),
        ...     Metric("alpaca", "avg_tokens_seen", 100, AggregationType.MEAN),
        ... ]
        >>>
        >>> batch2_metrics = [
        ...     Metric("alpaca", "tokens_seen", 100, AggregationType.SUM),
        ...     Metric("alpaca", "avg_tokens_seen", 100, AggregationType.MEAN),
        ... ]
        >>>
        >>> # Update with metrics
        >>> aggregator.update(batch1_metrics)
        >>> aggregator.update(batch2_metrics)
        >>>
        >>> # Get final results
        >>> results = aggregator.get_metrics_for_logging(prefix="train")
        >>> # {"train_alpaca/tokens_seen": 200.0, "train_alpaca/avg_tokens_seen": 100.0}

    Raises:
        ValueError: If dist_window_size is not positive.
    """

    def __init__(self, dist_window_size: int = 1000):
        if dist_window_size <= 0:
            raise ValueError(
                f"dist_window_size must be positive, got {dist_window_size}"
            )

        # Storage: {(dataset, metric): MetricState} - O(unique metrics) not O(samples)
        self._metric_states: dict[tuple[str, str], MetricState] = {}
        self._dist_window_size = dist_window_size

        # Track aggregation types for validation - prevents same metric name with different agg types
        self._metric_agg_types: dict[tuple[str, str], AggregationType] = {}

        # Create handler registry - all handlers initialized upfront
        self._handlers: dict[AggregationType, AggregationHandler] = {
            AggregationType.SUM: SumAggHandler(),
            AggregationType.MAX: MaxAggHandler(),
            AggregationType.MIN: MinAggHandler(),
            AggregationType.MEAN: MeanAggHandler(),
            AggregationType.DISTRIBUTION: DistributionAggHandler(dist_window_size),
            AggregationType.CATEGORICAL_COUNT: CategoricalCountAggHandler(),
        }

    def _validate_metric_consistency(self, metric: Union[Metric, MetricState]) -> None:
        """Validate that metric name uses consistent aggregation type."""
        metric_key = (metric.dataset_name, metric.metric_name)
        metric_name = metric.metric_name

        if metric_key in self._metric_agg_types:
            existing_agg_type = self._metric_agg_types[metric_key]
            if existing_agg_type != metric.agg_type:
                raise ValueError(
                    f"Metric '{metric_name}' in dataset '{metric.dataset_name}' "
                    f"is already registered with aggregation type {existing_agg_type.value}, "
                    f"but a handler or user code tried to use it with type {metric.agg_type.value}. "
                    f"Use different metric names for different aggregation types."
                )
        else:
            # Track this metric's aggregation type
            self._metric_agg_types[metric_key] = metric.agg_type

    def register_handler(
        self, agg_type: AggregationType, handler: AggregationHandler
    ) -> None:
        """Register custom aggregation handler for specified type.

        Args:
            agg_type (AggregationType): The aggregation type to handle
            handler (AggregationHandler): Handler instance implementing the AggregationHandler interface
        """
        # Warn if replacing a handler that's already in use
        if agg_type in self._handlers and any(
            state.agg_type == agg_type for state in self._metric_states.values()
        ):
            logger.warning(
                f"Replacing handler for {agg_type} - aggregation type already in use by existing metrics. "
                f"This may affect existing metric behavior."
            )

        self._handlers[agg_type] = handler

    def update(self, metrics: list[Metric]) -> None:
        """Update (dataset_name, metric_name) metric state with new values.

        Args:
            metrics (list[Metric]): List of metrics to update the state with

        Raises:
            ValueError: If no handler is registered for a metric's aggregation type,
                       or if metric name conflicts with existing aggregation type.
        """
        for metric in metrics:
            # Same metric name must use same aggregation type
            self._validate_metric_consistency(metric)

            metric_key = (metric.dataset_name, metric.metric_name)
            handler = self._handlers.get(metric.agg_type)

            if handler is None:
                raise ValueError(
                    f"No handler registered for aggregation type: {metric.agg_type}"
                )

            if metric_key not in self._metric_states:
                self._metric_states[metric_key] = handler.initialize_metric_state(
                    metric.dataset_name, metric.metric_name, metric.agg_type
                )

            local_agg_metric = self._metric_states[metric_key]
            handler.update(local_agg_metric, metric)  # Mutates local_agg_metric

    def get_metrics_for_logging(self, prefix: str = "data") -> dict[str, float]:
        """Get final metrics for logging in standard format.

        Args:
            prefix (str): Prefix for metric names in the returned dictionary

        Returns:
            dict[str, float]: Dictionary with keys like "{prefix}_{dataset_name}/{metric_name}"
                and float values. For example, with `prefix="train"`, `dataset_name="alpaca"`,
                `metric_name="loss"`, the key would be `train_alpaca/loss`.
        """
        final_results = self._compute_unified_metrics()

        return {
            f"{prefix}_{result.dataset_name}/{result.metric_name}": result.value
            for result in final_results
        }

    def _compute_unified_metrics(self) -> list[MetricState]:
        """
        Compute metrics handling both local and distributed cases uniformly.

        Returns:
            list[MetricState]: Final results ready for logging
        """
        # Step 1: Get local results from all handlers (may expand distributions/categoricals)
        prepared_results = []
        for local_agg_metric in self._metric_states.values():
            handler = self._handlers[local_agg_metric.agg_type]
            generated_metrics = handler.finalize_local_agg(local_agg_metric)

            # Validate each newly generated metric state immediately
            for gen_metric in generated_metrics:
                self._validate_metric_consistency(gen_metric)

            prepared_results.extend(generated_metrics)

        # Step 2: Apply distributed reduction if needed
        if dist.is_initialized() and dist.get_world_size() > 1:
            prepared_results = self._finalize_dist_agg(prepared_results)

        return prepared_results

    def _finalize_dist_agg(
        self, local_agg_metrics: list[MetricState]
    ) -> list[MetricState]:
        """Apply distributed reduction to local results.

        Args:
            local_agg_metrics (list[MetricState]): (dataset_name, metric_name) metric pairs from this rank

        Returns:
            list[MetricState]: Reduced results combining all ranks
        """
        world_size = dist.get_world_size()

        # Gather all results from all ranks
        all_results = [None] * world_size
        dist.all_gather_object(all_results, local_agg_metrics)

        # Group by (dataset_name, metric_name) for reduction
        grouped = defaultdict(list)
        for rank_results in all_results:
            if rank_results:  # Handle ranks with no metrics
                for result in rank_results:
                    result_key = (result.dataset_name, result.metric_name)
                    grouped[result_key].append(result)

        # Apply handler-specific distributed reduction
        reduced_results = []
        for result_key, results_list in grouped.items():
            if not results_list:
                continue  # Skip empty groups

            # All results for a key should have same agg_type
            agg_type = results_list[0].agg_type
            handler = self._handlers[agg_type]
            reduced_result = handler.finalize_dist_agg(results_list)
            reduced_results.append(reduced_result)

        return reduced_results

    def state_dict(self) -> dict[str, Any]:
        """Serialize aggregator state for checkpointing.

        Returns:
            dict[str, Any]: Serializable dictionary containing all aggregator state
        """
        serializable_state = {}
        required_agg_types = set()  # Track aggregation types used in saved states

        for metric_key, local_agg_metric in self._metric_states.items():
            # Get handler for this result's aggregation type
            handler = self._handlers[local_agg_metric.agg_type]
            required_agg_types.add(local_agg_metric.agg_type)

            # Convert MetricState to serializable dict
            result_dict = {
                "dataset_name": local_agg_metric.dataset_name,
                "metric_name": local_agg_metric.metric_name,
                "value": local_agg_metric.value,
                "agg_type": local_agg_metric.agg_type,
                "metadata": handler.serialize_metadata(local_agg_metric.metadata),
            }

            # Convert tuple key to string for JSON compatibility
            serializable_state[str(metric_key)] = result_dict

        return {
            "state": serializable_state,
            "dist_window_size": self._dist_window_size,
            "required_agg_types": list(
                required_agg_types
            ),  # Save which handlers are needed
            # Save which aggregation types are used for each metric
            "metric_agg_types": {
                str(k): v.value for k, v in self._metric_agg_types.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load aggregator state from checkpoint.

        Args:
            state_dict (dict[str, Any]): Dictionary containing serialized aggregator state

        Raises:
            ValueError: If required handlers are missing after checkpoint restore
        """
        self._dist_window_size = state_dict.get("dist_window_size", 1000)

        # Sanity check: Ensure all required handlers are available
        required_agg_types = state_dict.get("required_agg_types", [])
        missing_handlers = []
        for agg_type in required_agg_types:
            if agg_type not in self._handlers:
                missing_handlers.append(agg_type)

        if missing_handlers:
            raise ValueError(
                f"Missing handlers for aggregation types: {missing_handlers}. "
                f"Custom handlers must be re-registered before checkpoint restore."
            )

        deserialized_state = {}
        for key_str, result_dict in state_dict["state"].items():
            # Convert string keys back to tuples
            metric_key = ast.literal_eval(key_str)

            # Get handler for this aggregation type
            agg_type = result_dict["agg_type"]
            handler = self._handlers[agg_type]

            # Restore metadata using handler-specific deserialization
            metadata = handler.deserialize_metadata(result_dict["metadata"])

            # Create MetricState from dict
            local_agg_metric = MetricState(
                dataset_name=result_dict["dataset_name"],
                metric_name=result_dict["metric_name"],
                value=result_dict["value"],
                agg_type=result_dict["agg_type"],
                metadata=metadata,
            )

            deserialized_state[metric_key] = local_agg_metric

        self._metric_states = deserialized_state

        # Restore validation state
        self._metric_agg_types = {}
        for key_str, agg_type_str in state_dict.get("metric_agg_types", {}).items():
            key = ast.literal_eval(key_str)
            self._metric_agg_types[key] = AggregationType(agg_type_str)
