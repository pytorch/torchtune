# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class FullFinetuneParams:
    # Model
    model: str = ""
    model_checkpoint: str = ""

    # Tokenizer
    tokenizer: str = ""
    tokenizer_checkpoint: str = ""

    # Dataset and Sampler
    dataset: str = ""
    shuffle: bool = True
    batch_size: int = 2

    # Optimizer and Scheduler
    optimizer: str = "SGD"
    lr: float = 2e-5
    loss: str = "CrossEntropyLoss"

    # Training
    epochs: int = 3
    max_steps_per_epoch: Optional[int] = None
    resume_from_checkpoint: bool = False
    run_generation: Optional[int] = None

    # Distributed
    cpu_offload: bool = False
    enable_fsdp: bool = True
    enable_activation_checkpointing: bool = True

    # Environment
    device: str = "cuda"
    dtype: str = "fp32"
    seed: Optional[int] = None

    # Logging
    output_dir: str = "/tmp/full_finetune_output"
    metric_logger_type: str = "disk"
    project: Optional[str] = None

    def __post_init__(self):
        for param in fields(self):
            if getattr(self, param.name) == "":
                raise TypeError(f"{param.name} needs to be specified")
