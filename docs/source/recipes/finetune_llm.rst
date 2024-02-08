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
