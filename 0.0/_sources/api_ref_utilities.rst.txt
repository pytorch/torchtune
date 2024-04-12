=================
torchtune.utils
=================

.. currentmodule:: torchtune.utils


.. _checkpointing_label:

Checkpointing
-------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    FullModelHFCheckpointer
    FullModelMetaCheckpointer

.. _dist_label:

Distributed
-----------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    init_distributed
    get_world_size_and_rank

.. _mp_label:

Mixed Precision
---------------

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

.. autosummary::
    :toctree: generated/
    :nosignatures:

    memory.set_activation_checkpointing

.. _metric_logging_label:

Metric Logging
--------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    metric_logging.WandBLogger
    metric_logging.TensorBoardLogger
    metric_logging.StdoutLogger
    metric_logging.DiskLogger

Data
----

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
