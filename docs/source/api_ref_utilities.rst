=================
torchtune.utils
=================

.. currentmodule:: torchtune.utils


.. _checkpointing_label:

Checkpointing
-------------

TorchTune offers checkpointers to allow seamless transitioning between checkpoint formats for training and interoperability with the rest of the ecosystem. For a comprehensive overview of
checkpointing, please see the :ref:`checkpointing deep-dive <understand_checkpointer>`.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    FullModelHFCheckpointer
    FullModelMetaCheckpointer

.. _dist_label:

Distributed
-----------

Utilities for enabling and working with distributed training.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    FSDPPolicyType
    init_distributed
    get_world_size_and_rank
    get_full_finetune_fsdp_wrap_policy

.. _mp_label:

Reduced Precision
------------------

Utilities for working in a reduced precision setting.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_dtype
    list_dtypes

.. _ac_label:

Memory Management
-----------------

Utilities to reduce memory consumption during training.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    set_activation_checkpointing

.. _perf_profiling_label:

Performance and Profiling
-------------------------

TorchTune provides utilities to profile and debug the performance
of your finetuning job.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    profiler

.. _metric_logging_label:

Metric Logging
--------------

Various logging utilities.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    metric_logging.WandBLogger
    metric_logging.TensorBoardLogger
    metric_logging.StdoutLogger
    metric_logging.DiskLogger

Data
----

Utilities for working with data and datasets.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    padded_collate

.. _gen_label:


Miscellaneous
-------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    TuneRecipeArgumentParser
    get_logger
    get_device
    set_seed
    generate
