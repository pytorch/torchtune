# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

from torch.utils.data import DataLoader
from torchtune.data import padded_collate_sft
from torchtune.data._metrics import MetricsAggregator


def collate_with_metrics(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function that extracts metrics and uses padded_collate_sft for the rest."""
    all_metrics = []
    clean_batch = []
    for sample in batch:
        if "metrics" in sample:
            all_metrics.extend(sample.pop("metrics"))
        clean_batch.append(sample)

    if not clean_batch:
        return {"metrics": all_metrics}

    # Use torchtune's standard SFT collate function
    collated = padded_collate_sft(clean_batch)
    collated["metrics"] = all_metrics
    return collated


def generate_ckpt(
    dataloader: DataLoader,
    aggregator: MetricsAggregator,
    steps_before_checkpoint: int,
    steps_after_checkpoint: int,
    resume_dataloader: Optional[DataLoader] = None,
    resume_aggregator: Optional[MetricsAggregator] = None,
) -> dict[str, Any]:
    """
    Generates a checkpoint by running through data and saving checkpoint mid-stream.
    Optionally, a second dataloader and aggregator can be given to resume from ckpt
    and run steps_after_checkpoint to match the first one.

    Args:
        dataloader (DataLoader): The dataloader to test
        aggregator (MetricsAggregator): The metrics aggregator to use
        steps_before_checkpoint (int): Number of steps to run before saving checkpoint
        steps_after_checkpoint (int): Number of steps to run after checkpoint
        resume_dataloader (Optional[DataLoader]): Optional new dataloader to test resuming.
            If None, returns empty resumed_batches.
        resume_aggregator (Optional[MetricsAggregator]): Optional new aggregator to test resuming.
            If None, returns empty resumed_metrics.

    Returns:
        dict[str, Any]: Dict with batches/metrics from both pre and post checkpoint runs.
    """
    iterator = iter(dataloader)

    # Collect batches before and after checkpoint
    batches = []
    checkpoint_state = None
    metrics_at_checkpoint = {}

    total_steps = steps_before_checkpoint + steps_after_checkpoint

    for idx, batch in enumerate(iterator):
        batches.append(batch)

        # Process metrics
        if "metrics" in batch:
            aggregator.update(batch.pop("metrics"))

        # Save checkpoint state after steps_before_checkpoint
        if idx == steps_before_checkpoint - 1:  # -1 because idx is 0-based
            checkpoint_state = {
                "loader": dataloader.state_dict(),
                "aggregator": aggregator.state_dict(),
            }
            metrics_at_checkpoint = aggregator.get_metrics_for_logging(prefix="train")

        # Stop after total steps
        if idx == total_steps - 1:
            break

    # Split batches
    pre_checkpoint_batches = batches[:steps_before_checkpoint]
    post_checkpoint_batches = batches[steps_before_checkpoint:]

    # Resume with new instances if provided
    resumed_batches = []
    resumed_metrics = {}

    if (
        resume_dataloader is not None
        and resume_aggregator is not None
        and checkpoint_state is not None
    ):
        # Test resuming with new instances
        resume_dataloader.load_state_dict(checkpoint_state["loader"])
        resume_aggregator.load_state_dict(checkpoint_state["aggregator"])
        resume_iterator = iter(resume_dataloader)

        # Collect only the post-checkpoint batches when resuming
        for idx, batch in enumerate(resume_iterator):
            resumed_batches.append(batch)

            # Process metrics
            if "metrics" in batch:
                resume_aggregator.update(batch.pop("metrics"))

            # Stop after steps_after_checkpoint
            if idx == steps_after_checkpoint - 1:
                break

        resumed_metrics = resume_aggregator.get_metrics_for_logging(prefix="train")

    return {
        # Original run
        "pre_checkpoint_batches": pre_checkpoint_batches,
        "post_checkpoint_batches": post_checkpoint_batches,
        "metrics_at_checkpoint": metrics_at_checkpoint,
        "final_metrics": aggregator.get_metrics_for_logging(prefix="train"),
        # Resumed run
        "resumed_batches": resumed_batches,
        "resumed_metrics": resumed_metrics,
        # Internal state for loading - only if someone needs to manually load
        "_checkpoint_state": checkpoint_state,
    }
