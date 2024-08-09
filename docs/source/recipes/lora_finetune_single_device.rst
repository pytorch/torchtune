.. _lora_finetune_recipe_label:

=============================
LoRA Single Device Finetuning
=============================

This recipe supports finetuning on next-token prediction tasks using `LoRA <https://arxiv.org/abs/2106.09685>`_,
a technique to significantly reduce memory consumption during training whilst still maintaining competitive performance.

Interested in using this recipe? Check out some of our awesome tutorials to show off how it can be used:

* :ref:`Finetuning Llama2 with LoRA<lora_finetune_label>`
* :ref:`End-to-End Workflow with torchtune<dataset_tutorial_label>`
* :ref:`Fine-tuning Llama3 with Chat Data<chat_tutorial_label>`
* :ref:`Meta Llama3 in torchtune<llama3_label>`
* :ref:`Fine-Tune Your First LLM<finetune_llama_label>`

The best way to get started with our recipes is through the :ref:`cli_label`, which allows you to
list all our recipes and configs, run recipes, copy configs and recipes, and validate configs
without touching a line of code!

For example, if you're interested in using this recipe with the latest `Llama models <https://llama.meta.com/>`_, you can fine-tune
in just two steps:


.. note::

    You may need to be granted access to the Llama model you're interested in. See
    :ref:`here <download_llama_label>` for details on accessing gated repositories.


.. code-block:: bash

    tune download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output-dir /tmp/Meta-Llama-3.1-8B-Instruct \
    --ignore-patterns "original/consolidated.00.pth"

    tune run lora_finetune_single_device \
    --config llama3_1/8B_lora_single_device


Most of you will want to twist, pull, and bop all the different levers, buttons, and knobs we expose in our recipes. Check out our
:ref:`configs tutorial <config_tutorial_label>` to learn how to customize recipes to suit your needs.

This recipe is an example of :ref:`Parameter efficient fine-tuning (PEFT) <glossary_peft>`. To understand the different
levers you can pull, see our documentation for the different PEFT training paradigms we support:

* :ref:`glossary_lora`.
* :ref:`glossary_qlora`.

As with all of our recipes, you can also:

* Adjust :ref:`model precision <glossary_precision>`.
* Use :ref:`activation checkpointing <glossary_act_ckpt>`.
* Enable :ref:`gradient accumulation <glossary_grad_accm>`.
* Use :ref:`lower precision optimizers <glossary_low_precision_opt>`. However, note that since LoRA
  significantly reduces memory usage due to gradient state, you will likely not need this
  feature.

If you're interested in an overview of our memory optimization features, check out our  :ref:`memory optimization overview<memory_optimization_overview_label>`!
