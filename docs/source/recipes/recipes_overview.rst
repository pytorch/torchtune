.. _recipes_overview_label:

================
Recipes Overview
================

Recipes are the primary entry points for torchtune users.
These can be thought of as **hackable, singularly-focused scripts for interacting with LLMs** including fine-tuning,
inference, evaluation, and quantization.

Each recipe consists of three components:

* **Configurable parameters**, specified through yaml configs and command-line overrides
* **Recipe script**, entry-point which puts everything together including parsing and validating configs, setting up the environment, and correctly using the recipe class
* **Recipe class**, core logic needed for fine-tuning, exposed through a set of APIs

.. note::

  To learn more about the concept of "recipes", check out our technical deep-dive: :ref:`recipe_deepdive`.


Finetuning
----------

Our recipes include:

* :ref:`Single-device LoRA fine-tuning <lora_finetune_recipe_label>`.
* Single-device full fine-tuning
* Distributed full fine-tuning
* Distributed LoRA fine-tuning
* Direct Preference Optimization (DPO)
* Proximal Policy Optimization (PPO)
* :ref:`Distributed Quantization-Aware Training (QAT)<qat_distributed_recipe_label>`.

For a full list, please run:

.. code-block:: bash

    tune ls

.. Alignment finetuning
.. --------------------
.. Interested in alignment fine-tuning? You've come to the right place! We support the following alignment techniques:

.. Direct Preference Optimixation (DPO) Fine-Tuning
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. `Direct Preference Optimixation <https://arxiv.org/abs/2305.18290>`_ (DPO) stype techniques allow for aligning language models with respect
.. to a reward model objective function without the use of reinforcement learning. We support DPO preference fine-tuning with:

..   * :ref:`Single-device <lora_finetune_recipe_label>` and :ref:`multi-device <lora_finetune_recipe_label>` LoRA finetuning.

.. note::

  Our recipe documentation is currently in construction. Please feel free to follow the progress in our tracker
  issue `here <https://github.com/pytorch/torchtune/issues/1408>`_.
