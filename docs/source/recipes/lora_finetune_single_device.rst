.. _lora_finetune_recipe_label:

=============================
LoRA Single Device finetuning
=============================

This recipe supports finetuning on next-token prediction tasks using `LoRA <https://arxiv.org/abs/2106.09685>`_,
a technique to significantly reduce memory consumption during training whilst still maintaining competitive performance.

Interested in using this recipe? Check out some of our awesome tutorials to show off how it can be used:

* :ref:`Finetuning Llama2 with LoRA<lora_finetune_label>`
* :ref:`End-to-End Workflow with torchtune<dataset_tutorial_label>`
* :ref:`Fine-tuning Llama3 with Chat Data<chat_tutorial_label>`
* :ref:`Meta Llama3 in torchtune<llama3_label>`
* :ref:`Fine-Tune Your First LLM<finetune_llama_label>`

The best way to get started with our recipes is through the :ref:`cli_label`, which allows you to start fine-tuning
one of our built-in models without touching a single line of code!

For example, if you're interested in using this recipe with the latest `Llama models <https://llama.meta.com/>`_, you can fine-tune
in just two steps:


.. note::

    You may need to be granted access to the LLama model you're interested in. See
    :ref:`here <download_llama_label>` for details on accessing gated repositories.


.. code-block:: bash

    tune download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output-dir /tmp/Meta-Llama-3.1-8B-Instruct \
    --ignore-patterns "original/consolidated.00.pth"

    tune run lora_finetune_single_device \
    --config llama3_1/8B_lora_single_device

.. note::

    The :ref:`cli_label` allows you to list all our recipes and configs, run recipes, copy configs and recipes,
    and validate configs without touching a line of code!


Most of you will want to twist, pull, and bop all the different levers and knobs we expose in our recipes. Check out our
:ref:`configs tutorial <config_tutorial_label>` to learn how to customize recipes to suit your needs.

Are you also interested in our memory optimisation features? Check out our  :ref:`memory optimization overview<memory_optimisation_overview_label>`!
This recipe in particular supports :ref:`parameter efficient fine-tuning (PEFT) <glossary_peft>`: :ref:`glossary_lora` and :ref:`glossary_qlora`.

As with all our single-device recipes, you can also:

* Adjust :ref:`model precision <glossary_precision>`.
* Use :ref:`activation checkpointing <glossary_act_ckpt>`.
* Enable :ref:`gradient accumulation <glossary_grad_accm>`.
* Use :ref:`lower precision optimizers <glossary_low_precision_opt>`. However, note that since LoRA
  significantly reduces memory usage due to gradient state, you will likely not need this
  feature.
