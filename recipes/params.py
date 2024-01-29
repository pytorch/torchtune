# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional


@dataclass
class FullFinetuneParams:
    # Environment
    device: str
    dtype: str

    # Model
    model: str
    model_checkpoint: str

    # Tokenizer
    tokenizer: str
    tokenizer_checkpoint: str

    # Dataset and Sampler
    dataset: str
    shuffle: bool
    batch_size: int

    # Optimizer and Scheduler
    optimizer: str
    lr: float
    loss: str

    # Training
    epochs: int
    resume_from_previous_checkpoint: bool

    # Logging
    output_dir: str
    metric_logger_type: str

    # Distributed
    cpu_offload: bool = False
    activation_checkpointing: bool = False

    # Defaults
    seed: Optional[int] = None
    max_steps_per_epoch: Optional[int] = None
    project: Optional[str] = None
    run_generation: Optional[int] = None
