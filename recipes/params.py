# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List

@dataclass
class FullFinetuneParams:
    # Environment
    device: str
    dtype: str

    # Reproducability
    seed: int

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
    max_steps_per_epoch: int
    resume_from_checkpoint: bool

    # Logging
    output_dir: str

@dataclass
class LoRAFinetuneParams:
    # Environment
    device: str
    dtype: str

    # Reproducability
    seed: int

    # Model
    model: str
    model_checkpoint: str
    lora_attn_modules: List[str]

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
    max_steps_per_epoch: int
    resume_from_checkpoint: bool

    # Logging
    output_dir: str
