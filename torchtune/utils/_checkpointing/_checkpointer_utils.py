
from enum import Enum


class CheckpointFormat(Enum):
    META = "meta"
    HF = "hf"
    TORCHTUNE_NEW = "torchtune_new"
    TORCHTUNE_RESTART = "torchtune_restart"


class ModelType(Enum):
    LLAMA2 = "llama2"


def is_torchtune_checkpoint(checkpoint_format: CheckpointFormat) -> bool:
    return (
        checkpoint_format == CheckpointFormat.TORCHTUNE_NEW or
        checkpoint_format == CheckpointFormat.TORCHTUNE_RESTART
    )
