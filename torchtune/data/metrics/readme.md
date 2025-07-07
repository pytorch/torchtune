# TorchTune Metrics Module

## Overview

The metrics module provides a robust system for tracking and aggregating training metrics across multiple datasets and distributed environments. It follows a **strategy pattern** design with pluggable aggregation handlers to efficiently handle different types of metrics.

## Architecture Overview

```
┌────────────────────────────────────────────────────┐
│                Training Loop                       │
└─────────────────────┬──────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────┐
│                   MetricTransform                  │
│  • Applied to each sample                          │
│  • Generates per-sample metrics                    │
│  • Examples: tokens_seen, seq_len, samples_seen    │
└─────────────────────┬──────────────────────────────┘
                      │ list[Metric]
┌─────────────────────▼──────────────────────────────┐
│                 MetricsAggregator                  │
│  • Aggregates metrics across samples and ranks     │
│  • Uses pluggable AggregationHandlers              │
│  • Handles distributed reduction                   │
└─────────────────────┬──────────────────────────────┘
                      │ {prefix}_{dataset_name}/{metric_name} # prefix is "train", "val", etc.
┌─────────────────────▼──────────────────────────────┐
│                 Logging System                     │
│  • W&B, TensorBoard, etc.                          │
│  • Gets formatted metrics ready for logging        │
└────────────────────────────────────────────────────┘
```

## File Structure

- **`_metric_transform.py`**: Defines `Metric`, `AggregationType`, and transform classes
- **`_metric_agg_handlers.py`**: Aggregation strategy implementations
- **`_metric_aggregator.py`**: Main aggregator orchestrating the handlers

## Customizing metrics

- **Custom transforms**: Extend `MetricTransform` for domain-specific metrics
- **Handler registration**: Register custom handlers for specialized aggregation needs

#######
## TODO
## Move this from here to website docs
#######

## Core Components

### 1. MetricTransform
Generates per-sample metrics during data processing.

**Key Features:**
- Applied to each sample in the dataset
- Creates `Metric` objects with dataset name, metric name, value, and aggregation type
- Handles dataset namespacing for multi-dataset scenarios

**Example Usage:**
```python
from torchtune.data.metrics import DefaultTrainingMetricTransform, AggregationType

transform = DefaultTrainingMetricTransform()
transform.set_dataset_name("alpaca")

# Applied to each sample
sample = {"tokens": [1, 2, 3, 4, 5]}
sample = transform(sample)
# sample["metrics"] now contains:
# [
#   Metric(dataset_name="alpaca", name="samples_seen", value=1, agg_type=AggregationType.SUM),
#   Metric(dataset_name="alpaca", name="tokens_seen", value=5, agg_type=AggregationType.SUM),
#   Metric(dataset_name="alpaca", name="seq_len", value=5, agg_type=AggregationType.DISTRIBUTION)
# ]
```

### 2. MetricsAggregator
Efficiently aggregates metrics across samples and distributed ranks.

**Key Features:**
- Handler-based strategy pattern for different aggregation types
- Distributed-aware with automatic rank reduction
- Checkpointable state for training resumption
- Keep track of (metric, dataset) pairs

**Aggregation Types (at the time of writing):**
- `SUM`: Cumulative totals (e.g., total tokens processed)
- `MEAN`: Running averages (e.g., average loss)
- `MAX/MIN`: Extrema tracking (e.g., max sequence length seen)
- `DISTRIBUTION`: Statistical summaries (mean, min, max, percentiles)
- `CATEGORICAL_COUNT`: Category cumulative counts (e.g. num of samples from a given category)

**Example Usage:**
```python
from torchtune.data.metrics import MetricsAggregator, Metric, AggregationType

# Create aggregator
aggregator = MetricsAggregator()

# Sample metrics from different batches
batch1_metrics = [
    Metric("alpaca", "tokens_seen", 100, AggregationType.SUM),
    Metric("alpaca", "avg_tokens_seen", 100, AggregationType.MEAN),
]

batch2_metrics = [
    Metric("alpaca", "tokens_seen", 100, AggregationType.SUM),
    Metric("alpaca", "avg_tokens_seen", 100, AggregationType.MEAN),
]

# Update with metrics
aggregator.update(batch1_metrics)
aggregator.update(batch2_metrics)

# Get final results
results = aggregator.get_metrics_for_logging(prefix="train")
# {"train_alpaca/tokens_seen": 200.0, "train_alpaca/avg_tokens_seen": 100.0}
```

### 3. AggregationHandlers
Pluggable strategies for different aggregation patterns.

```
AggregationHandler (ABC)
├── SumAggHandler          # value += metric.value
├── MeanAggHandler         # tracks sum and count
├── MaxAggHandler          # value = max(value, metric.value)
├── MinAggHandler          # value = min(value, metric.value)
├── DistributionAggHandler # maintains value window + stats
└── CategoricalCountAggHandler # Counter for categories
```

**Custom Handler Example:**
```python
class CustomAggHandler(AggregationHandler):
    def initialize_metric_state(self, dataset_name, metric_name, agg_type):
        return MetricState(
            dataset_name=dataset_name,
            metric_name=metric_name,
            value=<initial_value>, # should change
            agg_type=agg_type,
            metadata={} # may need to change
        )

    def update(self, local_agg_metric, metric):
        ...

    def finalize_local_agg(self, local_agg_metric):
        ...

    def finalize_dist_agg(self, local_agg_metrics):
        ...

# Register with aggregator
aggregator.register_handler(AggregationType.CUSTOM, CustomAggHandler())
```

## Distributed Training Support

The metrics system automatically handles distributed environments:

1. **Local Aggregation**: Each rank aggregates its own metrics
2. **Distributed Reduction**: Results are combined across ranks using `all_gather_object`
3. **Type-Aware Reduction**: Each aggregation type uses appropriate reduction (sum, mean, max, etc.)

**Distributed Flow:**
```
Rank 0: [(ds1, metric1), (ds1, metric2)] → LocalAgg → [(ds1, metric1), (ds1, metric2)]
Rank 1: [(ds1, metric1), (ds1, metric2)] → LocalAgg → [(ds1, metric1), (ds1, metric2)]
                                ↓
                          AllGather + Reduce
                                ↓
                          Final Results [(ds1, metric1), (ds1, metric2)]
```
