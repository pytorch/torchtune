===============
torchtune.utils
===============

.. currentmodule:: torchtune.utils

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


.. _gen_label:


Miscellaneous
-------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_logger
    get_device
    generate
    torch_version_ge
