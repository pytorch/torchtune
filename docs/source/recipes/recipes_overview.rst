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


Supervised Finetuning
---------------------

torchtune provides built-in recipes for finetuning on single device, on multiple devices with `FSDP <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/>`_,
using a variety of :ref:`memory optimization features <memory_optimization_overview_label>`. Our  fine-tuning recipes support all of our models and all our dataset types.
This includes continued pre-training, and various supervised funetuning paradigms, which can be customized through our datasets. Check out our
:ref:`dataset tutorial <dataset_tutorial_label>` for more information.

Our supervised fine-tuning recipes include:

* :ref:`Single-device <lora_finetune_recipe_label>` LoRA fine-tuning.
* :ref:`Distributed Quantization-Aware Training<qat_distributed_recipe_label>`.

.. Alignment finetuning
.. --------------------
.. Interested in alignment fine-tuning? You've come to the right place! We support the following alignment techniques:

.. Direct Preference Optimixation (DPO) Fine-Tuning
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. `Direct Preference Optimixation <https://arxiv.org/abs/2305.18290>`_ (DPO) stype techniques allow for aligning language models with respect
.. to a reward model objective function without the use of reinforcement learning. We support DPO preference fine-tuning with:

..   * :ref:`Single-device <lora_finetune_recipe_label>` and :ref:`multi-device <lora_finetune_recipe_label>` LoRA finetuning.

.. note::

  Want to learn more about a certain recipe, but can't find the documentation here?
  Not to worry! Our recipe documentation is currently in construction - come back soon
  to see documentation of your favourite fine-tuning techniques.

  .. interested in contributing documentation? Check out our issue here TODO (SalmanMohammadi)
