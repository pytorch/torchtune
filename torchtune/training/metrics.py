# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch


def compute_classification_metrics(
    split,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    batch,
    tokenizer,
    running_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute metrics for the given logits and labels.
    """
    if "sample_weights" in batch:
        sample_weights = batch["sample_weights"].to(labels.device)
    else:
        # samples_weights are all 1 of size label
        sample_weights = torch.ones(labels.shape[0], device=labels.device)

    # Compute accuracy based on the next token predicted. We find this
    # by finding the last non-pad token in the sequence.
    # Note that we'll return -1 if no pad tokens are found, which is
    # the last item as we want.
    sequence_lengths = torch.eq(tokens, tokenizer.pad_id).int().argmax(-1) - 1
    # sequence_lengths = sequence_lengths % tokens.shape[-1]
    sequence_lengths = sequence_lengths.to(logits.device)

    logits = logits[
        torch.arange(logits.shape[0], device=logits.device),
        sequence_lengths,
    ]
    labels = labels[
        torch.arange(labels.shape[0], device=labels.device),
        sequence_lengths,
    ]

    predicted = torch.argmax(logits, dim=1)

    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    running_metrics["correct"] += correct
    running_metrics["total"] += total

    correct_weighted = ((predicted == labels).float() * sample_weights).sum().item()
    total_weighted = sample_weights.sum().item()
    running_metrics["correct_weighted"] += correct_weighted
    running_metrics["total_weighted"] += total_weighted

    predicted_binary_irrelevant = predicted.eq(
        torch.tensor(
            tokenizer.encode("irrelevant", add_bos=False, add_eos=False),
            device=labels.device,
        )
    )
    labels_binary_irrelevant = labels.eq(
        torch.tensor(
            tokenizer.encode("irrelevant", add_bos=False, add_eos=False),
            device=labels.device,
        )
    )

    tp = (predicted_binary_irrelevant * labels_binary_irrelevant).sum().item()
    tn = ((~predicted_binary_irrelevant) * (~labels_binary_irrelevant)).sum().item()
    fp = (predicted_binary_irrelevant * (~labels_binary_irrelevant)).sum().item()
    fn = ((~predicted_binary_irrelevant) * labels_binary_irrelevant).sum().item()

    running_metrics["tp"] += tp
    running_metrics["tn"] += tn
    running_metrics["fp"] += fp
    running_metrics["fn"] += fn

    # weighted versions
    tp_weighted = (
        (predicted_binary_irrelevant * labels_binary_irrelevant * sample_weights)
        .sum()
        .item()
    )
    tn_weighted = (
        ((~predicted_binary_irrelevant) * (~labels_binary_irrelevant) * sample_weights)
        .sum()
        .item()
    )
    fp_weighted = (
        (predicted_binary_irrelevant * (~labels_binary_irrelevant) * sample_weights)
        .sum()
        .item()
    )
    fn_weighted = (
        ((~predicted_binary_irrelevant) * labels_binary_irrelevant * sample_weights)
        .sum()
        .item()
    )

    running_metrics["tp_weighted"] += tp_weighted
    running_metrics["tn_weighted"] += tn_weighted
    running_metrics["fp_weighted"] += fp_weighted
    running_metrics["fn_weighted"] += fn_weighted

    if split != "":
        split = f"{split}_"

    metrics_dict = {
        f"{split}batch_accuracy": correct / total,
        f"{split}running_accuracy": running_metrics["correct"]
        / running_metrics["total"],
        f"{split}running_precision_I": (
            0
            if (running_metrics["tp"] + running_metrics["fp"]) == 0
            else running_metrics["tp"] / (running_metrics["tp"] + running_metrics["fp"])
        ),
        f"{split}running_recall_I": (
            0
            if (running_metrics["tp"] + running_metrics["fn"]) == 0
            else running_metrics["tp"] / (running_metrics["tp"] + running_metrics["fn"])
        ),
        f"{split}running_accuracy_weighted": running_metrics["correct_weighted"]
        / running_metrics["total_weighted"],
        f"{split}running_precision_I_weighted": (
            0
            if (running_metrics["tp_weighted"] + running_metrics["fp_weighted"]) == 0
            else running_metrics["tp_weighted"]
            / (running_metrics["tp_weighted"] + running_metrics["fp_weighted"])
        ),
        f"{split}running_recall_I_weighted": (
            0
            if (running_metrics["tp_weighted"] + running_metrics["fn_weighted"]) == 0
            else running_metrics["tp_weighted"]
            / (running_metrics["tp_weighted"] + running_metrics["fn_weighted"])
        ),
    }

    return (metrics_dict, running_metrics)
