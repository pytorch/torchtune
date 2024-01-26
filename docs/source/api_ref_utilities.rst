=================
torchtune.utils
=================

.. currentmodule:: torchtune.utils

.. _dist_label:

Distributed
------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    distributed.init_distributed
    distributed.get_world_size_and_rank
    distributed.get_fsdp

.. _mp_label:

Mixed Precsion
--------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    precision.get_autocast
    precision.get_gradient_scaler
    precision.get_dtype
    precision.list_dtypes

.. _ac_label:

Memory Management
-----------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    memory.set_activation_checkpointing

Metric Logging
--------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    metric_logging.get_metric_logger
    metric_logging.WandBLogger
    metric_logging.TensorBoardLogger
    metric_logging.StdoutLogger
    metric_logging.DiskLogger

Data
-----

.. autosummary::
    :toctree: generated/
    :template: class.rst

    checkpointable_dataloader.CheckpointableDataLoader
    collate.padded_collate

Checkpoint saving & loading
----------------------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    checkpoint.save_checkpoint
    checkpoint.load_checkpoint

.. _gen_label:

Generation
----------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    generation.GenerationUtils
    generation.generate_from_prompt


Miscellaneous
-------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    argparse.TuneArgumentParser
    logging.get_logger
    device.get_device
    seed.set_seed
