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


**Ccontinued pre-training** and **supervised fine-tuning** paradigms are widely supported in our recipes through customising
the dataset you're working with. Check out our dataset tutorial for more information :ref:`data_set_tutorial_label`.

* :ref:`LoRA Single Device finetuning <lora_finetune_recipe_label>`
* :ref:`LoRA Multi-Device finetuning <lora_finetune_recipe_label>`
* :ref:`Full-memory Single Device finetuning <lora_finetune_recipe_label>`
* :ref:`Full-memory Multi-Device finetuning <lora_finetune_recipe_label>`



Interested in alignment fine-tuning? You've come to the right place! We support the following alignment techniques:

* Single and multi-device, LoRA, :ref:`reward-model-free, finetuning <lora_finetune_recipe_label>` techniques based on
  `Direct Preference Optimisation <https://arxiv.org/abs/2305.18290>`_.
