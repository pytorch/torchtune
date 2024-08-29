# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ._checkpointing import (  # noqa
    ADAPTER_CONFIG,
    ADAPTER_KEY,
    Checkpointer,
    EPOCHS_KEY,
    FullModelHFCheckpointer,
    FullModelMetaCheckpointer,
    FullModelTorchTuneCheckpointer,
    MAX_STEPS_KEY,
    MODEL_KEY,
    ModelType,
    OPT_KEY,
    RNG_KEY,
    SEED_KEY,
    STEPS_KEY,
    TOTAL_EPOCHS_KEY,
    update_state_dict_for_classifier,
)

from ._generation import generate, generate_next_token  # noqa
from ._profiler import (
    DEFAULT_PROFILE_DIR,
    DEFAULT_PROFILER_ACTIVITIES,
    DEFAULT_SCHEDULE,
    DEFAULT_TRACE_OPTS,
    DummyProfiler,
    PROFILER_KEY,
    setup_torch_profiler,
)
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
from .pooling import get_unmasked_sequence_lengths

from .precision import get_dtype, set_default_dtype, validate_expected_param_dtype
from .seed import set_seed

__all__ = [
    "update_state_dict_for_classifier",
    "get_memory_stats",
    "FSDPPolicyType",
    "log_memory_stats",
    "get_dtype",
    "get_logger",
    "get_unmasked_sequence_lengths",
    "set_activation_checkpointing",
    "set_default_dtype",
    "set_seed",
    "validate_expected_param_dtype",
    "torch_version_ge",
    "OptimizerInBackwardWrapper",
    "create_optim_in_bwd_wrapper",
    "register_optim_in_bwd_hooks",
    "DEFAULT_PROFILE_DIR",
    "DEFAULT_PROFILER_ACTIVITIES",
    "DEFAULT_SCHEDULE",
    "DEFAULT_TRACE_OPTS",
    "DummyProfiler",
    "PROFILER_KEY",
    "setup_torch_profiler",
    "generate",
    "generate_next_token",
]
