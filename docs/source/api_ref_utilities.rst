===============
torchtune.utils
===============

.. currentmodule:: torchtune.utils

.. _dist_label:

Distributed
-----------

Utilities for enabling and working with distributed training.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    FSDPPolicyType
    init_distributed
    is_distributed
    get_world_size_and_rank
    get_full_finetune_fsdp_wrap_policy
    lora_fsdp_wrap_policy

.. _ac_label:

Memory Management
-----------------

Utilities to reduce memory consumption during training.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    set_activation_checkpointing
    OptimizerInBackwardWrapper
    create_optim_in_bwd_wrapper
    register_optim_in_bwd_hooks


Performance and Profiling
-------------------------

torchtune provides utilities to profile and debug the memory and performance
of your finetuning job.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_memory_stats
    log_memory_stats

.. _gen_label:


Miscellaneous
-------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_logger
    get_device
    set_seed
    generate
    torch_version_ge
