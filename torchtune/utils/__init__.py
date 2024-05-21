# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ._checkpointing import (  # noqa
    FullModelHFCheckpointer,
    FullModelMetaCheckpointer,
    FullModelTorchTuneCheckpointer,
    ModelType,
    transform_opt_state_dict,
)

from ._device import get_device
from ._distributed import (  # noqa
    contains_fsdp,
    FSDPPolicyType,
    get_full_finetune_fsdp_wrap_policy,
    get_world_size_and_rank,
    init_distributed,
    is_distributed,
    lora_fsdp_wrap_policy,
    prepare_model_for_fsdp_with_meta_device,
    validate_no_params_on_meta_device,
)
from ._generation import generate
from ._profiler import profiler
from .argparse import TuneRecipeArgumentParser
from .collate import padded_collate, padded_collate_dpo
from .constants import (  # noqa
    ADAPTER_KEY,
    EPOCHS_KEY,
    MAX_STEPS_KEY,
    MODEL_KEY,
    OPT_KEY,
    SEED_KEY,
    TOTAL_EPOCHS_KEY,
)
from .logging import get_logger
from .memory import (  # noqa
    cleanup_before_training,
    create_optim_in_bwd_wrapper,
    get_memory_stats,
    log_memory_stats,
    OptimizerInBackwardWrapper,
    register_optim_in_bwd_hooks,
    set_activation_checkpointing,
)

from .precision import (
    get_dtype,
    list_dtypes,
    set_default_dtype,
    validate_expected_param_dtype,
)
from .quantization import get_quantizer_mode
from .seed import set_seed

__all__ = [
    "transform_opt_state_dict",
    "get_memory_stats",
    "FSDPPolicyType",
    "log_memory_stats",
    "get_device",
    "get_dtype",
    "get_logger",
    "get_world_size_and_rank",
    "init_distributed",
    "is_distributed",
    "list_dtypes",
    "lora_fsdp_wrap_policy",
    "get_full_finetune_fsdp_wrap_policy",
    "padded_collate",
    "padded_collate_dpo",
    "set_activation_checkpointing",
    "set_default_dtype",
    "set_seed",
    "validate_expected_param_dtype",
    "TuneRecipeArgumentParser",
    "OptimizerInBackwardWrapper",
    "create_optim_in_bwd_wrapper",
    "register_optim_in_bwd_hooks",
    "profiler",
    "get_quantizer_mode",
    "generate",
]
