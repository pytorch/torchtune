.. _lora_finetune_recipe_label:

=============================
LoRA Single Device finetuning
=============================

This recipe supports finetuning using `LoRA <https://arxiv.org/abs/2106.09685>`_, a technique to significantly reduce memory consumption during training
whilst still maintaining competitive performance.

Interested in using this recipe? Check out some of our awesome tutorials to show off how it can be used:

* :ref:`Finetuning Llama2 with LoRA<lora_finetune_label>`
* :ref:`End-to-End Workflow with torchtune<dataset_tutorial_label>`
* :ref:`Fine-tuning Llama3 with Chat Data<chat_tutorial_label>`
* :ref:`Meta Llama3 in torchtune<llama3_label>`
* :ref:`Fine-Tune Your First LLM<finetune_llama_label>`

We provide the following configurations out-of-the box. To learn how to customize recipes to suit your needs, check out
our :ref:`configs tutorial <config_tutorial_label>`:

.. code-block:: bash

    lora_finetune_single_device          llama2/7B_lora_single_device
                                         llama2/7B_qlora_single_device
                                         code_llama2/7B_lora_single_device
                                         code_llama2/7B_qlora_single_device
                                         llama3/8B_lora_single_device
                                         llama3/8B_qlora_single_device
                                         llama2/13B_qlora_single_device
                                         mistral/7B_lora_single_device
                                         mistral/7B_qlora_single_device
                                         gemma/2B_lora_single_device
                                         gemma/2B_qlora_single_device
                                         gemma/7B_lora_single_device
                                         gemma/7B_qlora_single_device
                                         phi3/mini_lora_single_device
                                         phi3/mini_qlora_single_device

.. note::
    The :ref:`cli_label` allows you to :ref:`list <cli_label>`, :ref:`run <cli_label>`, :ref:`copy <cli_label>`,
    and :ref:`validate <cli_label>` configs without touching a line of code!


-- Salman: Just copy-pasted the recipe docs, to tidy up. broad strokes here.

LoRA finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe is optimized
  for single GPU training. Training on CPU is not supported.

  Features:
      - Activation Checkpointing. This can be controlled using the ``activation_checkpointing``
          flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
          activations in memory and instead recompute them during the backward pass. This is especially
          helpful for larger batch sizes when you're memory constrained. But these savings in memory
          come at the cost of training performance. In most cases training can slow-down quite a bit as
          a result of this activation recomputation.

      - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
          flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
          most cases this should halve the memory footprint of full precision (fp32) training, without
          loss in model quality (will depend on the model, training data and other settings). For
          GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
          precision are currently not supported.g

      - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
          controlled using the ``gradient_accumulation_steps`` flag.

              Total Batch Size = batch_size * gradient accumulation steps.

          For example: with batch_size=1 and gradient_accumulation_steps=32 we get a total batch size of 32.

          Gradient accumulation is especially useful when you are memory constrained. In this case,
          accumulating gradients might give you better training speed than enabling activation
          checkpointing.

      - Lower precision optimizers. This recipe supports lower-precision optimizers from the bitsandbytes
          library (https://huggingface.co/docs/bitsandbytes/main/en/index). We've tested the recipe with
          8-bit AdamW and Paged AdamW.

      - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
          training. Currently we checkpoint both the adapter weights (trainable params only) and the
          complete merged weights (adapter weights added back to the base model). For more details
          please take a look at our LoRA tutorial
          (https://pytorch.org/torchtune/main/tutorials/lora_finetune.html).

          Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
          only saved at the end of a given epoch and used in case of resuming training. Resuming
          training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
          currently not supported.

          For more details on the checkpointer, please take a look at
          our checkpointer deepdive (https://pytorch.org/torchtune/main/tutorials/checkpointer.html).

      - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

  For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
  has example commands for how to kick-off training.
