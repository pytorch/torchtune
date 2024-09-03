===============
torchtune.utils
===============

.. currentmodule:: torchtune.utils

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

    get_device
    get_logger
    generate
    torch_version_ge
