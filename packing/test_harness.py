# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
from fifo_bins_packing import OnTheFlyPackedDataset as FIFOBinsPackedDataset
from fifo_buffer_packing import OnTheFlyPackedDataset as FIFOBufferPackedDataset

# Import your three packing implementations as separate scripts
# For simplicity, assume they are saved as sort_packing.py, fifo_buffer_packing.py, fifo_bins_packing.py
from sort_packing import OnTheFlyPackedDataset as SortPackedDataset
from torchtune.datasets import alpaca_dataset
from torchtune.models.llama3 import llama3_tokenizer


def run_packing_strategy(
    max_seq_len, padding_idx, buffer_size, strategy_name, num_bins=None
):
    """Run a packing strategy and compute statistics."""

    if strategy_name == "Sort Min-Max":
        dataset = alpaca_dataset(tokenizer=tokenizer, packed=False, split="train[:5%]")
        packed_dataset = SortPackedDataset(
            dataset, max_seq_len, padding_idx, buffer_size
        )
    elif strategy_name == "FIFO with Buffer":
        dataset = alpaca_dataset(tokenizer=tokenizer, packed=False, split="train[:5%]")
        packed_dataset = FIFOBufferPackedDataset(
            dataset, max_seq_len, padding_idx, buffer_size
        )
    elif strategy_name == "FIFO with Bins":
        dataset = alpaca_dataset(tokenizer=tokenizer, packed=False, split="train[:5%]")
        packed_dataset = FIFOBinsPackedDataset(
            dataset, max_seq_len, padding_idx, buffer_size, num_bins=num_bins
        )
    elif strategy_name == "Torchtune Packed":
        dataset = alpaca_dataset(tokenizer=tokenizer, packed=True, split="train[:5%]")
    elif strategy_name == "Torchtune Not Packed":
        packed_dataset = alpaca_dataset(
            tokenizer=tokenizer, packed=False, split="train[:5%]"
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    iterator = iter(packed_dataset)
    pack_count = 0
    total_padding_count = 0
    total_tokens_count = 0
    start_time = time.time()

    for pack in iterator:
        pack_count += 1
        tokens = pack["tokens"]
        padding_count = (
            tokens.count(padding_idx)
            if isinstance(tokens, list)
            else (tokens == padding_idx).sum().item()
        )
        total_padding_count += padding_count
        total_tokens_count += len(tokens)

    total_time = time.time() - start_time
    avg_time_per_pack = total_time / pack_count if pack_count > 0 else 0
    padding_percentage = (
        (total_padding_count / total_tokens_count) * 100
        if total_tokens_count > 0
        else 0
    )

    stats = {
        "Total packs generated": pack_count,
        "Total processing time (s)": round(total_time, 2),
        "Average time per pack (s)": round(avg_time_per_pack, 4),
        "Overall padding percentage": round(padding_percentage, 2),
    }
    if strategy_name not in ["Torchtune Packed", "Torchtune Not Packed"]:
        custom_stats = iterator._compute_stats()
        stats.update(
            {
                "Maximum age of used sequences (packs)": custom_stats[
                    "Maximum age of used sequences (packs)"
                ],
                "Average age of used sequences (packs)": custom_stats[
                    "Average age of used sequences (packs)"
                ],
            }
        )

    return stats


if __name__ == "__main__":

    # Test parameters
    max_seq_lens = [1024, 2048, 4096, 8192]
    padding_idx = 0
    buffer_size = 100  # Fixed buffer size for consistency
    num_bins = 50  # For FIFO with bins
    strategies = [
        "Sort Min-Max",
        "FIFO with Buffer",
        "FIFO with Bins",
        "Torchtune Packed",
        "Torchtune Not Packed",
    ]

    for max_seq_len in max_seq_lens:
        print(f"\nTesting with max_seq_len={max_seq_len}")
        print("=" * 50)

        # Tokenizer setup
        tokenizer = llama3_tokenizer(
            path="/tmp/Llama-3.2-1B-Instruct/original/tokenizer.model",
            max_seq_len=max_seq_len,
        )

        for strategy in strategies:
            print(f"\nStrategy: {strategy}")

            stats = run_packing_strategy(
                max_seq_len,
                padding_idx,
                buffer_size,
                strategy,
                num_bins=num_bins,
            )
            for key, value in stats.items():
                print(f"{key}: {value}")
            print("-" * 50)
