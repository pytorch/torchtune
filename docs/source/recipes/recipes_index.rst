.. _recipes_index_label:

==============
Recipes Index
==============

Recipes are the primary entry points for torchtune users.
These can be thought of as **hackable, singularly-focused scripts for interacting with LLMs** including training,
inference, evaluation, and quantization.

Each recipe consists of three components:

* **Configurable parameters**, specified through yaml configs and command-line overrides
* **Recipe script**, entry-point which puts everything together including parsing and validating configs, setting up the environment, and correctly using the recipe class
* **Recipe class**, core logic needed for training, exposed through a set of APIs

.. note::

  To learn more about the concept of "recipes", check out our technical deep-dive: :ref:`recipe_deepdive`.

torchtune provides built-in recipes for finetuning on single device, on multiple devices with `FSDP <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/>`_,
using memory efficient techniques like `LoRA <https://arxiv.org/abs/2106.09685>`_, and more!

.. note::

    You can also utilize the :code:`tune ls` command to print out all recipes and corresponding configs.

    .. code-block:: bash

        $ tune ls
        RECIPE                                   CONFIG
        full_finetune_single_device              llama2/7B_full_low_memory
                                                mistral/7B_full_low_memory
        full_finetune_distributed                llama2/7B_full
                                                llama2/13B_full
                                                mistral/7B_full
        lora_finetune_single_device              llama2/7B_lora_single_device
                                                llama2/7B_qlora_single_device
                                                mistral/7B_lora_single_device
        ...

Continued pre-training/Supervised finetuning
--------------------------------------------

Our  fine-tuning recipes support all of our models and all our dataset types [1]. This includes continued pre-training, and various supervised fune-tuning
paradigms, which can be customised through our datasets. Check out our :ref:`dataset tutorial <dataset_tutorial_label>` for more information. Our fine-tuning recipes
include:

* :ref:`Single-device <lora_finetune_recipe_label>` and :ref:`multi-device <lora_finetune_recipe_label>` full-finetuning.
* :ref:`Single-device <lora_finetune_recipe_label>` and :ref:`multi-device <lora_finetune_recipe_label>` LoRA-finetuning.

Alignment finetuning
--------------------

Interested in alignment fine-tuning? You've come to the right place! We support the following alignment techniques:

Reward modelling
^^^^^^^^^^^^^^^^

* Reward model training with:

  * :ref:`Single-device <lora_finetune_recipe_label>` and :ref:`multi-device <lora_finetune_recipe_label>` full-finetuning.
  * :ref:`Single-device <lora_finetune_recipe_label>` and :ref:`multi-device <lora_finetune_recipe_label>` LoRA finetuning.


Reinforcement Learning from Human Feedback (RLHF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :ref:`Preference finetuning <lora_finetune_recipe_label>` techniques based on `Direct Preference Optimisation <https://arxiv.org/abs/2305.18290>`_ with:

  * :ref:`Single-device <lora_finetune_recipe_label>` and :ref:`multi-device <lora_finetune_recipe_label>` LoRA finetuning.

* :ref:`Single-device <lora_finetune_recipe_label>` RLHF with Proximal Policy Optimisation full-finetuning.


.. note::

  [1] Our reward and preference modelling recipes currently support a limited subset of tasks.
  See the :class:`preference datasets <torchtune.datasets.PreferenceDataset>` page for more details.
