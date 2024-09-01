# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ._device import get_device
from ._distributed import (  # noqa
    contains_fsdp,
    FSDPPolicyType,
    get_full_finetune_fsdp_wrap_policy,
    get_full_model_state_dict,
    get_full_optimizer_state_dict,
    get_world_size_and_rank,
    init_distributed,
    is_distributed,
    load_from_full_model_state_dict,
    load_from_full_optimizer_state_dict,
    lora_fsdp_wrap_policy,
    prepare_model_for_fsdp_with_meta_device,
    set_torch_num_threads,
    shard_model,
    validate_no_params_on_meta_device,
)
from ._generation import generate, generate_next_token  # noqa

from ._version import torch_version_ge
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

from .seed import set_seed

__all__ = [
    "update_state_dict_for_classifier",
    "get_memory_stats",
    "FSDPPolicyType",
    "log_memory_stats",
    "get_device",
    "get_logger",
    "get_world_size_and_rank",
    "init_distributed",
    "is_distributed",
    "lora_fsdp_wrap_policy",
    "get_full_finetune_fsdp_wrap_policy",
    "set_activation_checkpointing",
    "set_seed",
    "torch_version_ge",
    "OptimizerInBackwardWrapper",
    "create_optim_in_bwd_wrapper",
    "register_optim_in_bwd_hooks",
    "generate",
    "generate_next_token",
    "shard_model",
]
