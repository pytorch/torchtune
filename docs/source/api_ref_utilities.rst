=================
torchtune.utils
=================

.. currentmodule:: torchtune.utils

.. _dist_label:

Distributed
------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    distributed.init_distributed
    distributed.get_world_size_and_rank
    distributed.get_fsdp

.. _mp_label:

Mixed Precsion
--------------

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

Metric Logging
--------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    metric_logging.get_metric_logger
    metric_logging.WandBLogger
    metric_logging.TensorBoardLogger
    metric_logging.StdoutLogger
    metric_logging.DiskLogger

Data
-----

.. autosummary::
    :toctree: generated/
    :nosignatures:

    checkpointable_dataloader.CheckpointableDataLoader
    collate.padded_collate

Checkpoint saving & loading
----------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    checkpoint.save_checkpoint
    checkpoint.load_checkpoint

.. _gen_label:

Generation
----------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    generation.GenerationUtils
    generation.generate_from_prompt


Miscellaneous
-------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    argparse.TuneArgumentParser
    logging.get_logger
    device.get_device
    seed.set_seed
