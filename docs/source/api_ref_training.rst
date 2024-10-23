==================
torchtune.training
==================

.. currentmodule:: torchtune.training

.. _checkpointing_label:

Checkpointing
-------------

torchtune offers checkpointers to allow seamless transitioning between checkpoint formats for training and interoperability with the rest of the ecosystem. For a comprehensive overview of
checkpointing, please see the :ref:`checkpointing deep-dive <understand_checkpointer>`.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    FullModelHFCheckpointer
    FullModelMetaCheckpointer
    FullModelTorchTuneCheckpointer
    ModelType
    FormattedCheckpointFiles
    update_state_dict_for_classifier

.. _mp_label:

Reduced Precision
------------------

Utilities for working in a reduced precision setting.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_dtype
    set_default_dtype
    validate_expected_param_dtype
    get_quantizer_mode

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

    apply_selective_activation_checkpointing
    set_activation_checkpointing
    OptimizerInBackwardWrapper
    create_optim_in_bwd_wrapper
    register_optim_in_bwd_hooks

.. _lr_scheduler_label:

Schedulers
----------

Utilities to control lr during the training process.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_cosine_schedule_with_warmup
    get_lr

.. _metric_logging_label:

Metric Logging
--------------

Various logging utilities.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    metric_logging.CometLogger
    metric_logging.WandBLogger
    metric_logging.TensorBoardLogger
    metric_logging.StdoutLogger
    metric_logging.DiskLogger

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
    setup_torch_profiler

Miscellaneous
-------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_unmasked_sequence_lengths
    set_seed
