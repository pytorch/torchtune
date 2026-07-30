# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reproduce the README optimization-flags table.

Runs `tune run full_finetune_single_device` (baseline) and `lora_finetune_single_device`
(QLoRA final row) on a Llama 3.2 3B model, turning each optimization flag on
sequentially, and prints peak GPU memory and tokens-per-second for every row.

This script is intended to be run on a single A100 (or equivalent) GPU with the
Llama 3.2 3B Instruct weights already downloaded under
``/tmp/Llama-3.2-3B-Instruct`` (override with ``--checkpoint-dir``).

Usage:
    python scripts/bench_optimization_flags.py
    python scripts/bench_optimization_flags.py --checkpoint-dir /data/Llama-3.2-3B-Instruct

The output is a markdown table matching the README "Optimization flags" section.
"""

import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


CONFIG_FULL = "llama3_2/3B_full_single_device"
CONFIG_QLORA = "llama3_2/3B_qlora_single_device"

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Row:
    name: str
    recipe: str
    config: str
    cli_overrides: list[str]


BASE_OVERRIDES = [
    "tokenizer.max_seq_len=4096",
    "gradient_accumulation_steps=1",
    "epochs=1",
    "batch_size=2",
]


def build_rows() -> list[Row]:
    rows = [
        Row("Baseline", "full_finetune_single_device", CONFIG_FULL,
            ["dataset.packed=False", "compile=False",
             f"loss=torchtune.modules.loss.LinearCrossEntropyLoss",
             "enable_activation_checkpointing=False",
             "optimizer_in_bwd=True", "enable_activation_offloading=False"]),
        Row("+ Packed Dataset", "full_finetune_single_device", CONFIG_FULL,
            ["dataset.packed=True", "compile=False",
             f"loss=torchtune.modules.loss.LinearCrossEntropyLoss",
             "enable_activation_checkpointing=False",
             "optimizer_in_bwd=True", "enable_activation_offloading=False"]),
        Row("+ Compile", "full_finetune_single_device", CONFIG_FULL,
            ["dataset.packed=True", "compile=True",
             f"loss=torchtune.modules.loss.LinearCrossEntropyLoss",
             "enable_activation_checkpointing=False",
             "optimizer_in_bwd=True", "enable_activation_offloading=False"]),
        Row("+ Linear Cross Entropy", "full_finetune_single_device", CONFIG_FULL,
            ["dataset.packed=True", "compile=True",
             f"loss=torchtune.modules.loss.LinearCrossEntropyLoss",
             "enable_activation_checkpointing=False",
             "optimizer_in_bwd=True", "enable_activation_offloading=False"]),
        Row("+ Activation Checkpointing", "full_finetune_single_device", CONFIG_FULL,
            ["dataset.packed=True", "compile=True",
             f"loss=torchtune.modules.loss.LinearCrossEntropyLoss",
             "enable_activation_checkpointing=True",
             "optimizer_in_bwd=True", "enable_activation_offloading=False"]),
        Row("+ Fuse optimizer step into backward",
            "full_finetune_single_device", CONFIG_FULL,
            ["dataset.packed=True", "compile=True",
             f"loss=torchtune.modules.loss.LinearCrossEntropyLoss",
             "enable_activation_checkpointing=True",
             "optimizer_in_bwd=False", "enable_activation_offloading=False"]),
        Row("+ Activation Offloading", "full_finetune_single_device", CONFIG_FULL,
            ["dataset.packed=True", "compile=True",
             f"loss=torchtune.modules.loss.LinearCrossEntropyLoss",
             "enable_activation_checkpointing=True",
             "optimizer_in_bwd=False", "enable_activation_offloading=True"]),
        Row("+ 8-bit AdamW", "full_finetune_single_device", CONFIG_FULL,
            ["dataset.packed=True", "compile=True",
             f"loss=torchtune.modules.loss.LinearCrossEntropyLoss",
             "enable_activation_checkpointing=True",
             "optimizer_in_bwd=False", "enable_activation_offloading=True",
             "optimizer=bitsandbytes.optim.PagedAdamW8bit"]),
        Row("LoRA", "lora_finetune_single_device", CONFIG_QLORA,
            ["dataset.packed=True", "compile=True",
             f"loss=torchtune.modules.loss.LinearCrossEntropyLoss",
             "enable_activation_checkpointing=True",
             "optimizer_in_bwd=False", "enable_activation_offloading=True"]),
        Row("QLoRA", "lora_finetune_single_device", CONFIG_QLORA,
            ["dataset.packed=True", "compile=True",
             f"loss=torchtune.modules.loss.LinearCrossEntropyLoss",
             "enable_activation_checkpointing=True",
             "optimizer_in_bwd=False", "enable_activation_offloading=True"]),
    ]
    return rows


def run_row(row: Row, checkpoint_dir: str | None) -> tuple[float, float]:
    overrides = list(BASE_OVERRIDES)
    if checkpoint_dir:
        overrides.append(f"checkpointer.checkpoint_dir={checkpoint_dir}")
    overrides.extend(row.cli_overrides)
    cmd = [
        sys.executable, "-m", "torchtune.cli.tune", "run",
        row.recipe, "--config", row.config, *overrides,
    ]
    print(f"\n### {row.name}\n$ {' '.join(shlex.quote(c) for c in cmd)}\n",
          flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    peak_mem = _parse_metric(out, r"peak[_ ]?memory[^0-9]*([\d.]+)\s*GiB",
                             r"max[_ ]?memory.*?([\d.]+)\s*GiB")
    tps = _parse_metric(out, r"tokens[_ ]?per[_ ]?second[^0-9]*([\d.]+)",
                        r"tps[^0-9]*([\d.]+)")
    return peak_mem, tps


def _parse_metric(text: str, *patterns: str) -> float:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return float("nan")


def fmt_table(results: list[tuple[Row, float, float]]) -> str:
    lines = ["| Technique | Peak Memory Active (GiB) | % Change Memory vs Previous | Tokens Per Second | % Change Tokens/sec vs Previous |",
             "|:--|:-:|:-:|:-:|:-:|"]
    prev_mem = prev_tps = None
    for row, mem, tps in results:
        dmem = _pct(mem, prev_mem)
        dtps = _pct(tps, prev_tps)
        lines.append(f"| {row.name} | {mem} | {dmem} | {tps} | {dtps} |")
        prev_mem, prev_tps = mem, tps
    return "\n".join(lines)


def _pct(cur: float, prev: float | None) -> str:
    if prev is None:
        return "-"
    if prev == 0:
        return "-"
    return f"{(cur - prev) / prev * 100:+.2f}%"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", default=None,
                   help="Local path to Llama-3.2-3B-Instruct weights")
    args = p.parse_args()

    rows = build_rows()
    results = []
    for row in rows:
        try:
            mem, tps = run_row(row, args.checkpoint_dir)
        except Exception as exc:
            print(f"!! {row.name} failed: {exc}", file=sys.stderr)
            mem = tps = float("nan")
        results.append((row, mem, tps))

    print("\n\n=== README table (regenerated) ===\n")
    print(fmt_table(results))


if __name__ == "__main__":
    main()
