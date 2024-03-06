
from enum import Enum


class CheckpointFormat(Enum):
    META_FORMAT = "meta_format"
    HF_FORMAT = "hf_format"
    TORCHTUNE_FORMAT = "torchtune_format"


class ModelType(Enum):
    LLAMA2_7B = "llama2_7b"
