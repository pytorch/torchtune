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
    tokenizer,
    running_metrics: Dict[str, Any],
    sample_weights: torch.Tensor = None,
) -> Dict[str, Any]:
    """
    Compute metrics for the given logits and labels.
    """
    if sample_weights is not None:
        sample_weights = sample_weights.to(labels.device)
    else:
        # samples_weights are all 1 of size label
        sample_weights = torch.ones(labels.shape[0], device=labels.device)

    # For accuracy, do exact token match per batch for all positions where the label is not -100 or eos_id

    # predicted = torch.argmax(logits, dim=1)
    predicted = torch.argmax(logits, dim=2)
    valid_mask = (labels != -100) & (labels != tokenizer.eos_id)
    correct_mask = ((predicted == labels) & valid_mask).int().max(dim=1)[0]

    correct = correct_mask.sum().item()

    # correct = (predicted == labels).sum().item()
    total = labels.size(0)
    running_metrics["correct"] += correct
    running_metrics["total"] += total

    # correct_weighted = ((predicted == labels).float() * sample_weights).sum().item()
    correct_weighted = (correct_mask.float() * sample_weights).sum().item()
    total_weighted = sample_weights.sum().item()
    running_metrics["correct_weighted"] += correct_weighted
    running_metrics["total_weighted"] += total_weighted

    # Compute binary precision recall based on the next token predicted.
    # With llama3, words are tokenized into subwords. For simplicity, we'll just use the first token of the label.
    # We need to pull a fixed sequence out instead of
    # only using the last token.

    # We find the last non-pad token in the sequence.
    # Note that we'll return -1 if no pad tokens are found, which is
    # the last item as we want.
    sequence_lengths = torch.eq(tokens, tokenizer.pad_id).int().argmax(-1) - 1
    # sequence_lengths = sequence_lengths % tokens.shape[-1]
    sequence_lengths = sequence_lengths.to(logits.device)

    # For simplicity, we'll just use the first token of the irrelevant encoded
    irrelevant_encoded = torch.tensor(
        tokenizer.encode("irrelevant", add_bos=False, add_eos=False)[0],
        device=logits.device,
    )

    # Create tensor of positions to pull out per batch
    # positions = torch.stack([
    #     torch.arange(sl, sl+len(irrelevant_encoded)) for sl in sequence_lengths
    # ], dim=0)

    predicted = predicted[
        torch.arange(predicted.shape[0], device=logits.device),
        sequence_lengths,
    ]
    labels = labels[
        torch.arange(labels.shape[0], device=labels.device),
        sequence_lengths,
    ]

    predicted_binary_irrelevant = predicted.eq(irrelevant_encoded)
    labels_binary_irrelevant = labels.eq(irrelevant_encoded)

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
