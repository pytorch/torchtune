# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from enum import Enum
from typing import Dict

import torch


# Only mean metric reduction supported right now
class _ReductionType(Enum):
    MEAN = "mean"


import defaultdict


# LIMITATIONS
# 1) only support for CPU recording. Cannot be used to record GPU ops.
# 2) Only mean reduction is supported, and mean is computed on the fly.
# 3) Multiple reductions for a specific metric are not supported.
# 4) No multi-threaded support.


@dataclass
class _MetricTacker:
    count: int = 0
    val: float = 0.0


def _compute_rolling_average(prev_avg, val, n) -> float:
    return prev_avg * (n - 1) / n + val / n


# Only support for one reduction type. And the same metric can't have multiple reduction.


class TunePerfMonitor:
    def __init__(self):
        self._metric_dict: Dict[str, _MetricTacker] = defaultdict(_MetricTacker)
        self._inflight: Dict[str, float] = {}

    def _run_rolling_average_reduction(
        self, metric_name: str, metric_val: float
    ) -> float:
        # Rolling average
        metric_info = self._metric_dict[metric_name]
        count, prev_avg = metric_info.count + 1, metric_info.val
        avg = _compute_rolling_average(prev_avg=prev_avg, val=metric_val, n=count)
        return avg

    def _run_reduction(
        metric_name: str, metric_val: float, reduction_type: _ReductionType
    ) -> float:
        # TODO: if/else on reduction_type is not scalable. Update to using a dispatch
        # when we have more reduction types.
        if self.reduction_type == _ReductionType.MEAN:
            if metric_name not in self._metric_dict:
                return metric_val
            else:
                return self._run_rolling_average_reduction(
                    metric_name=metric_name, metric_val=metric_val
                )

        raise RuntimeError(f"Reduction type {reduction_type} not supported!")

    def _record_metric(slef, metric_name, metric_val):
        """
        Updates our metric tracking for metric_name with metric_val and bumps count by 1.
        """
        if metric_name not in self._metric_dict:
            self._metric_dict[metric_name] = _MetricTacker(count=1, val=metric_val)
        else:
            self._metric_dict[metric_name].count += 1
            self._metric_dict[metric_name].val = metric_val

    def start_record(metric: str):
        """
        Start tracking metric given by ``str``
        """
        assert (
            metric not in self._inflight
        ), f"Already tracking metric {metric} for this iteration!"
        self._inflight[metric] = time.perf_counter()

    def end_record(
        metric_name: str, reduction_type: _ReductionType = _ReductionType.MEAN
    ):
        """
        End tracking metric given by ``str`` and compute and store the metric value.
        """
        # TODO: only MEAN reduction supported right now, and end_record is not the best place to specify it.
        # Don't want to specify it in emit_metric either, as then we'll have to store metric for every
        # iteration, possibly too much space used. Could have a solution where user a priori defines the metric
        # they want to track and the reduction type for it.
        assert (
            metric_name in self._inflight
        ), f"Not tracking metric {metric_name} for this iteration!"
        cpu_elapsed_s = time.perf_counter() - self._inflight.pop(metric_name)

        # Compute metric and add to metric_dict.
        reduced_metric = self._run_reduction(metric_name, cpu_elapsed_s, reduction_type)
        self._record_metric(metric, reduced_metric)

    def log_metric(self, metric_name, metric_val):
        """
        Logs a metric at a single point in time. Use this for metrics that don't require
        a start/end (like QPS does), but instead to record a value at a specific point in time
        (like torch.cuda.max_memory_allocated())
        """
        reduced_metric = self._run_reduction(
            metric_name, metric_val, reduction_type=_ReductionType.MEAN
        )
        self._record_metric(metric_name, reduced_metric)

    def get_metric_val(metric_name, default: float) -> float:
        """
        Get the current value of metric given by metric_name. Note that the metric is already reduced with the reduction
        specified in ``end_record``.
        """
        # TODO: its pretty limiting to only specify the reduction in end_record.
        if metric_name not in self.metric_dict:
            return default
        return self.metric_dict[metric_name].val
