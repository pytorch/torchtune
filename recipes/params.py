from dataclasses import dataclass

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

    #Tokenizer
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
