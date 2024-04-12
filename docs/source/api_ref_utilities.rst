=================
torchtune.utils
=================

.. currentmodule:: torchtune.utils


.. _checkpointing_label:

Checkpointing
-------------

TorchTune offers checkpointers to allow seamless transitioning between checkpoint formats for training and interoperability with the rest of the ecosystem. For a comprehensive overview of
checkpointing, please see the :ref:`checkpointing tutorial <understand_checkpointer>`.

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

    init_distributed
    get_world_size_and_rank

.. _mp_label:

Reduced Precision
------------------

Utilities for working in a reduced precision setting.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    precision.get_autocast
    precision.get_gradient_scaler
    precision.get_dtype
    precision.list_dtypes

.. _ac_label:

Memory Management
-----------------

Utilities to reduce memory consumption during training.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    memory.set_activation_checkpointing

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

    collate.padded_collate

.. _gen_label:


Miscellaneous
-------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    argparse.TuneRecipeArgumentParser
    logging.get_logger
    get_device
    seed.set_seed
