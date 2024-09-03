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

    setup_torch_profiler

Miscellaneous
-------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_unmasked_sequence_lengths
