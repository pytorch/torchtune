# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Optional, Protocol, Union


class AggregationType(Enum):
    """Defines how a metric's value should be aggregated."""

    SUM = "sum"
    MEAN = "mean"
    DISTRIBUTION = "distribution"
    CATEGORICAL_COUNT = "categorical_count"
    MAX = "max"
    MIN = "min"


@dataclass(frozen=True)
class Metric:
    """A self-describing metric object."""

    dataset_name: str
    name: str
    value: Union[int, float, str]
    agg_type: AggregationType


class MetricTransform(Protocol):
    """Protocol for metric transforms."""

    def set_dataset_name(self, dataset_name: str) -> None: ...
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]: ...


class StandardMetricTransform(MetricTransform):
    """
    Attaches per-sample metrics for tracking training progress.

    This transform is responsible for generating metrics on a per-sample
    basis (e.g., tokens per sample). The actual aggregation of these metrics
    (eg calculating sum of samples seen) is handled by the
    `MetricsAggregator`. This separation of concerns ensures that metrics are
    correctly aggregated even with multiple dataloader workers and in a
    distributed setting.

    Tracked metrics include:
    - samples_seen: A count of samples processed.
    - tokens_seen: The cumulative sum of all tokens processed.
    - seq_len: A distribution of sequence lengths.
    """

    def __init__(self):
        # dataset_name is set by the dataset using set_dataset_name
        self.dataset_name: Optional[str] = None
        self.new_metric: Optional[Callable] = None

    def set_dataset_name(self, dataset_name: str) -> None:
        """Called by dataset to set the namespace for metrics.
        The dataset name is used to differentiate multiple datasets stats,
        e.g. "train/dataset1/tokens_seen" and "train/dataset2/tokens_seen"."""
        self.dataset_name = dataset_name
        self.new_metric = partial(Metric, dataset_name=dataset_name)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.dataset_name is None or self.new_metric is None:
            raise RuntimeError(
                "set_dataset_name() must be called before using the transform."
            )

        # Determine token key
        token_key = "tokens" if "tokens" in sample else "input_ids"
        token_len = len(sample.get(token_key, []))

        # Create metrics for this sample
        metrics = [
            self.new_metric(name="samples_seen", value=1, agg_type=AggregationType.SUM),
            self.new_metric(
                name="tokens_seen", value=token_len, agg_type=AggregationType.SUM
            ),
            self.new_metric(
                name="seq_len", value=token_len, agg_type=AggregationType.DISTRIBUTION
            ),
        ]

        # Append to existing metrics list or create new one
        if "metrics" not in sample:
            sample["metrics"] = []
        sample["metrics"].extend(metrics)
        return sample 