# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

from torchtune.datasets import ALL_DATASETS
from torchtune.models import ALL_MODELS, ALL_TOKENIZERS
from torchtune.utils.metric_logging import ALL_METRIC_LOGGERS
from torchtune.utils.precision import PRECISION_STR_TO_DTYPE


@dataclass
class FullFinetuneParams:
    """Arguments for the finetune_llm recipe.

    Args:
        device (str): Device to use for training. Options are "cpu" and "cuda"
        dtype (str): Data type to use for training.
        seed (int): Random seed to use for training.
        model (str): String specifying model architecture to fine-tune. See ``torchtune.models.get_model`` for options.
        model_checkpoint (str): Local path to load model checkpoint from.
        tokenizer (str): String specifying tokenizer to use. See ``torchtune.models.get_tokenizer`` for options.
        tokenizer_checkpoint (str): Local path to load tokenizer checkpoint from.
        dataset (str): String specifying dataset to use. See ``torchtune.datasets.get_dataset`` for options.
            Currently, only predefined datasets in library are supported.
        shuffle (bool): Whether to shuffle dataset.
        batch_size (int): Batch size to use for training.
        epochs (int): Number of epochs to train for.
        optimizer (str): String specifying optimizer to use. See ``torchtune.optim.get_optimizer`` for options.
        loss (str): String specifying loss function to use. See ``torchtune.losses.get_loss`` for options.
        lr (float): Learning rate to use for optimizer.
        activation_checkpointing (bool): Whether to use activation checkpointing.
        output_dir (str): Local path to save checkpoints and logs to.
        run_generation (int): Run eval on a prompt every ``run_generation`` steps. Set to 0 to disable.
        max_steps_per_epoch (int): Maximum number of steps to take per epoch.
        metric_logger_type (str): String specifying metric logger to use. See ``torchtune.utils.get_metric_logger``
            for options.
        project (str): Project name to use for logging. Used by ``WandBLogger``.
        resume_from_previous_checkpoint (bool): Whether to resume fine-tuning from a previous checkpoint.
        cpu_offload (bool): Whether to offload model to CPU.

    Raises:
        ValueError: If ``cpu_offload`` is ``True`` but ``device`` is not ``cuda`` and <= 1 GPUs.
    """

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

    def __post_init__(self):
        if self.cpu_offload and self.device != "cuda":
            raise ValueError(
                "Cannot offload model to CPU if device is not cuda or <= 1 GPUs."
            )
        if self.model not in ALL_MODELS:
            raise ValueError(
                f"Model not recognized. Expected one of {ALL_MODELS}, received {self.model}."
            )
        if self.tokenizer not in ALL_TOKENIZERS:
            raise ValueError(
                f"Tokenizer not recognized. Expected one of {ALL_TOKENIZERS}, received {self.tokenizer}."
            )
        if self.dataset not in ALL_DATASETS:
            raise ValueError(
                f"Dataset not recognized. Expected one of {ALL_DATASETS}, received {self.dataset}."
            )
        if self.metric_logger_type not in ALL_METRIC_LOGGERS:
            raise ValueError(
                f"Metric logger not recognized. Expected one of {ALL_METRIC_LOGGERS}, received {self.metric_logger_type}."
            )
        if self.dtype not in PRECISION_STR_TO_DTYPE:
            raise ValueError(
                f"Dtype {self.dtype} must be one of {', '.join(PRECISION_STR_TO_DTYPE.keys())} for finetuning."
            )
