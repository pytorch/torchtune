==================
torchtune.training
==================

.. currentmodule:: torchtune.training

Reduced Precision
------------------

Utilities for working in a reduced precision setting.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_quantizer_mode

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

.. _perf_profiling_label:

Performance and Profiling
-------------------------

torchtune provides utilities to profile and debug the memory and performance
of your finetuning job.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_memory_stats
    log_memory_stats
