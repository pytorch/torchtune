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

Miscellaneous
-------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_device
