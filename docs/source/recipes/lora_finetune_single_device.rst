.. _lora_finetune_recipe_label:

=============================
LoRA Single Device Finetuning
=============================

This recipe supports finetuning on next-token prediction tasks using parameter efficient fine-tuning techniques (PEFT)
such as :ref:`glossary_lora` and :ref:`glossary_qlora`. These techniques
significantly reduce memory consumption during training whilst still maintaining competitive performance.

We provide configs which you can get up and running quickly. Here is an example with llama 3.1 8B:

.. note::

    You may need to be granted access to the Llama model you're interested in. See
    :ref:`here <download_llama_label>` for details on accessing gated repositories.


.. code-block:: bash

    # download the model
    tune download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output-dir /tmp/Meta-Llama-3.1-8B-Instruct \
    --ignore-patterns "original/consolidated.00.pth"

    # run the recipe
    tune run lora_finetune_single_device \
    --config llama3_1/8B_lora_single_device

You can customize this recipe through the :ref:`cli_label`. For example, when fine-tuning with LoRA, you can adjust the layers which LoRA are applied to:

.. code-block:: bash

    tune run lora_finetune_single_device \
    --config llama3_1/8B_lora_single_device \
    model.lora_attn_modules=“[q_proj,k_proj,v_proj]” \
    model.apply_lora_to_mlp=True \
    model.lora_rank=64 \
    model.lora_alpha=128


For a deeper understanding of the different levers you can pull when using this recipe,
see our documentation for the different PEFT training paradigms we support:

* :ref:`glossary_lora`
* :ref:`glossary_qlora`

Many of our other memory optimization features can be used in this recipe. You can learn more about all of our memory optimization features in our :ref:`memory optimization overview<memory_optimization_overview_label>`.

Interested in seeing this recipe in action? Check out some of our tutorials to show off how it can be used:

* :ref:`Finetuning Llama2 with LoRA<lora_finetune_label>`
* :ref:`Finetuning Llama2 with QLoRA<qlora_finetune_label>`
* :ref:`End-to-End Workflow with torchtune<dataset_tutorial_label>`
* :ref:`Fine-tuning Llama3 with Chat Data<chat_tutorial_label>`
* :ref:`Meta Llama3 in torchtune<llama3_label>`
* :ref:`Fine-Tune Your First LLM<finetune_llama_label>`
