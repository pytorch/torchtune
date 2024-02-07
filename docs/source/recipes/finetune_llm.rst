.. _basic_finetune_llm:

=====================
LLM Finetuning Recipe
=====================

.. note::
    Ensure that the documentation version matches the installed TorchTune version

This recipe is used for fine tuning language models on a given dataset. Given a model and dataset it'll train the model on the example text using cross entropy loss.

This recipe supports:

* :ref:`Mixed Precsion Training<mp_label>`

* :ref:`Distributed Training with FSDP<dist_label>`

* :ref:`Activation Checkpointing<ac_label>`

To run the recipe directly, launch with

.. code-block:: bash

    tune finetune_llm --config <finetune_config>

Recipe
------

Copy the recipe directly into your own script or notebook to modify and edit for yourself.

.. literalinclude:: ../../../recipes/finetune_llm.py

Configs
-------

.. tabs::

    .. tab:: alpaca_llama2_finetune

        .. literalinclude:: ../../../recipes/configs/alpaca_llama2_finetune.yaml
