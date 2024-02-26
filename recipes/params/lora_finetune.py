# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, fields
from typing import List, Optional

from torchtune.datasets import ALL_DATASETS
from torchtune.models import ALL_MODELS, ALL_TOKENIZERS
from torchtune.utils.metric_logging import ALL_METRIC_LOGGERS
from torchtune.utils.precision import PRECISION_STR_TO_DTYPE


@dataclass
class LoRAFinetuneParams:
    """Arguments for the finetune_lora recipe. Note that LoRA is currently only supported
    for attention modules (i.e. Q, K, V, output projections), and not for MLP layers.

    Args:
        model (str): String specifying model architecture to fine-tune. See ``torchtune.models.get_model`` for options.
        model_checkpoint (str): Local path to load model checkpoint from.
        lora_attn_modules (List[str]): List of attention modules to use for LoRA. Supported values are
            ["q_proj", "k_proj", "v_proj", "output_proj"].
        lora_rank (int): Rank of LoRA decompositions.
        lora_alpha (float): Alpha parameter for LoRA.
        lora_checkpoint (str): Local path to load LoRA weights from.
        tokenizer (str): String specifying tokenizer to use. See ``torchtune.models.get_tokenizer`` for options.
        tokenizer_checkpoint (str): Local path to load tokenizer checkpoint from.
        dataset (str): String specifying dataset to use. See ``torchtune.datasets.get_dataset`` for options.
            Currently, only predefined datasets in library are supported.
        train_on_input (bool): Whether to train on the prompt in addition to the response.
        use_clean (bool): Whether to use cleaned version of Alpaca dataset or not.
        shuffle (bool): Whether to shuffle dataset.
        batch_size (int): Batch size to use for training.
        epochs (int): Number of epochs to train for.
        optimizer (str): String specifying optimizer to use. See ``torchtune.optim.get_optimizer`` for options.
        weight_decay (float): Weight decay to use for optimizer.
        lr (float): Base learning rate rate to use for optimizer.
        lr_scheduler (str): String specifying learning rate scheduler to use. See
            ``torchtune.lr_schedulers.get_lr_scheduler`` for options.
        num_warmup_steps (int): Number of warmup steps to use for learning rate scheduler.
        loss (str): String specifying loss function to use. See ``torchtune.losses.get_loss`` for options.
        epochs (int): Number of epochs to train for.
        max_steps_per_epoch (int): Maximum number of steps to take per epoch.
        resume_from_checkpoint (bool): Whether to resume fine-tuning from a previous checkpoint.
        cpu_offload (bool): Whether to offload model to CPU.
        enable_fsdp (bool): Whether to use FSDP.
        enable_activation_checkpointing (bool): Whether to use activation checkpointing.
        device (str): Device to use for training. Options are "cpu" and "cuda"
        dtype (str): Data type to use for training.
        seed (int): Random seed to use for training.
        output_dir (str): Local path to save checkpoints and logs to.
        metric_logger_type (str): String specifying metric logger to use. See ``torchtune.utils.get_metric_logger``
            for options.
        project (str): Project name to use for logging. Used by ``WandBLogger``.
        log_every_n_steps (int): How often to log metrics.
    """

    # Model
    model: str = ""
    model_checkpoint: str = ""
    lora_attn_modules: List[str] = field(default_factory=list)
    lora_rank: int = 8
    lora_alpha: float = 16
    lora_checkpoint: Optional[str] = None

    # Tokenizer
    tokenizer: str = ""
    tokenizer_checkpoint: str = ""

    # Dataset and Sampler
    dataset: str = ""
    train_on_input: bool = True
    use_clean: bool = True
    shuffle: bool = True
    batch_size: int = 2

    # Optimizer and Scheduler
    optimizer: str = "AdamW"
    weight_decay: float = 0.01
    lr: float = 3e-4
    lr_scheduler: str = "cosine_with_warmup"
    num_warmup_steps: int = 100
    loss: str = "CrossEntropyLoss"

    # Training
    epochs: int = 1
    max_steps_per_epoch: Optional[int] = None
    resume_from_checkpoint: bool = False

    # Distributed
    cpu_offload: bool = False
    enable_fsdp: bool = False
    enable_activation_checkpointing: bool = False

    # Environment
    device: str = "cuda"
    dtype: str = "fp32"
    seed: Optional[int] = None

    # Logging
    output_dir: str = "/tmp/lora_finetune_output"
    metric_logger_type: str = "disk"
    project: Optional[str] = None
    log_every_n_steps: Optional[int] = None

    def __post_init__(self):
        for param in fields(self):
            if getattr(self, param.name) == "":
                raise TypeError(f"{param.name} needs to be specified")

        if self.cpu_offload and self.device != "cuda":
            raise ValueError(
                "Cannot offload model to CPU if device is not cuda or <= 1 GPUs."
            )
        if self.enable_fsdp and self.device == "cpu":
            raise ValueError("FSDP is not supported on CPU.")
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
        if len(self.lora_attn_modules) == 0:
            raise ValueError("Must specify at least one module to apply LoRA to")
