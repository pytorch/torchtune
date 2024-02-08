.. _basic_finetune_llm:

==========================
LLM Full Finetuning Recipe
==========================

The full fine-tune recipe updates all of the parameters of the model using supervised learning.
Given a model and a dataset comprising of input-label pairs, we train the model on these pairs using cross-entropy loss.

.. note::

  Full Fine-tuning is usually a lot more expensive that parameter-efficient techniques like LoRA, but
  in most cases results in a higher quality model.


This recipe supports:

* :ref:`Mixed Precsion Training<mp_label>`

* :ref:`Distributed Training with FSDP<dist_label>`

* :ref:`Activation Checkpointing<ac_label>`

This guide will walk you through the different aspects of the `recipe <https://github.com/pytorch-labs/torchtune/blob/main/recipes/full_finetune.py>`_.


An example config for training the Llama 7B model using the Alpaca dataset looks something like this:

.. code-block:: yaml

    # Dataset and Dataloader
    dataset: alpaca
    seed: null
    shuffle: True

    # Model Arguments
    model: llama2_7b
    model_checkpoint: /tmp/llama2-7b
    tokenizer: llama2_tokenizer
    tokenizer_checkpoint: /tmp/tokenizer.model

    # Fine-tuning arguments
    batch_size: 2
    lr: 2e-5
    epochs: 3
    optimizer: SGD
    loss: CrossEntropyLoss
    output_dir: /tmp/alpaca-llama2-finetune
    device: cuda
    dtype: fp32
    enable_fsdp: True
    enable_activation_checkpointing: True
    resume_from_checkpoint: False

To run the recipe without any changes on 4 GPUs, launch a training run using TuneCLI:

.. code-block:: bash

    tune --nnodes 1 --nproc_per_node 4 --config alpaca_llama2_full_finetune

Dataset
-------

In this example, we use the `Alpaca Dataset <https://github.com/pytorch-labs/torchtune/blob/main/torchtune/datasets/alpaca.py>`_
from Stanford. The following parameters are related to the data:

.. code-block:: python

    # Point the dataset to the Alpaca Dataset implementation in TorchTune
    # This is set in the config
    dataset: alpaca

    # Don't mask the prompt during training
    # This is the default value
    train_on_input: True

    # Train on the raw data, not the cleaned version
    # This is the default value
    use_clean: False

    # Shuffle the data between epochs
    # This is set in the config
    shuffle: True

.. note::
    Shuffling the data after every epoch is a good practice. This helps makes sure the model does not learn
    spurious patterns related to the how the data is sequenced.

.. note::
    Set ``train_on_input`` to False if you want to learn on the label only i.e. mask out the prompt. The resulting loss
    will go down a lot slower.



Model
-----

In this example, we use the `Llama 7B model <https://github.com/pytorch-labs/torchtune/blob/main/torchtune/models/llama2.py>`_.
The following parameters are related to the model:

.. code-block:: python

    # Point the model to the default llama-7B model
    model: llama2_7b
    model_checkpoint: <PATH_TO_MODEL_CHECKPOINT>

    # Point to the default tokenizer for llama2
    tokenizer: llama2_tokenizer
    tokenizer_checkpoint: <PATH_TO_MODEL_TOKENIZER>

    # FSDP and Activation checkpointing are enabled
    enable_fsdp: True
    enable_activation_checkpointing: True


Training
--------

.. code-block:: python

    # Batch size refers to "local" batch size; global batch size is computed as
    # batch_size * num_gpus * gradient_accumulation_steps
    batch_size: 2
    lr: 2e-5
    epochs: 3

    optimizer: SGD

    epochs: 3
    loss: CrossEntropyLoss

    # default value corresponds to no accumulation
    gradient_accumulation_steps: 1

    # resume_from_checkpoint controls how the checkpoint is loaded at the beginning
    # of training; set this to True if a previously incomplete training is
    # restarting
    resume_from_checkpoint: False


.. note::
    The default optimizer is SGD instead of Adam since this uses less memory. Adam is known to result in better model
    quality.


And that's it! For more information on configs and how to update them, see this tutorial on Configs. For more information on recipes
see the tutorial on :ref:`Training Recipe Deep-Dive<recipe_deepdive>`
