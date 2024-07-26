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


.. code-block:: yaml

    # ANNOTATED YAML FILE
    # ANNOTATED YAML FILE
    # ANNOTATED YAML FILE
    # ANNOTATED YAML FILE
    # Tokenizer
    tokenizer:
      _component_: torchtune.models.gemma.gemma_tokenizer
      path: /tmp/gemma-2b/tokenizer.model

    # Dataset
    dataset:
      _component_: torchtune.datasets.alpaca_dataset
    seed: null
    shuffle: True

    # Model Arguments
    model:
      _component_: torchtune.models.gemma.lora_gemma_2b
      lora_attn_modules: ['q_proj', 'k_proj', 'v_proj']
      apply_lora_to_mlp: True
      lora_rank: 64
      lora_alpha: 16

    checkpointer:
      _component_: torchtune.utils.FullModelHFCheckpointer
      checkpoint_dir: /tmp/gemma-2b/
      checkpoint_files: [
        model-00001-of-00002.safetensors,
        model-00002-of-00002.safetensors,
      ]
      recipe_checkpoint: null
      output_dir: /tmp/gemma-2b
      model_type: GEMMA
    resume_from_checkpoint: False

    optimizer:
      _component_: torch.optim.AdamW
      lr: 2e-5

    lr_scheduler:
      _component_: torchtune.modules.get_cosine_schedule_with_warmup
      num_warmup_steps: 100

    loss:
      _component_: torch.nn.CrossEntropyLoss

    # Fine-tuning arguments
    batch_size: 4
    epochs: 3
    max_steps_per_epoch: null
    gradient_accumulation_steps: 1

    # Training env
    device: cuda

    # Memory management
    enable_activation_checkpointing: True

    # Reduced precision
    dtype: bf16

    # Logging
    metric_logger:
      _component_: torchtune.utils.metric_logging.DiskLogger
      log_dir: ${output_dir}
    output_dir: /tmp/alpaca-gemma-lora
    log_every_n_steps: 1
    log_peak_memory_stats: False
