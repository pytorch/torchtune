.. _finetune_llama_intel_gpu_label:

=======================================
Fine-Tune Llama3.1 on Intel GPU
=======================================

This guide will walk you through the process of launching your first finetuning
job on an Intel GPU, using Llama3.1 as example.


.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * Download a model from the `Hugging Face Hub <https://huggingface.co/docs/hub/en/index>`_
      * Modify a recipe's parameters to work on Intel GPUs
      * Run a fine-tuning job

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with the :ref:`overview of torchtune<overview_label>`
      * Make sure to :ref:`install torchtune<install_label>`

|

Set up your environment 
-----------------------

Please checkout `Getting Started On Intel GPU <https://pytorch.org/docs/stable/notes/get_start_xpu.html>`_ to setup environment.

|

Downloading a Llam3.1-8B-Instruct model
---------------------------------------
For this tutorial, we will be using the instruction-tuned version of Llama3.1-8B. First, let's download the model from Hugging Face. You will need to follow the instructions
on the `official Meta page <https://github.com/meta-llama/llama3/blob/main/README.md>`_ to gain access to the model.
Next, make sure you grab your Hugging Face token from `here <https://huggingface.co/settings/tokens>`_.


.. code-block:: bash

  tune download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output-dir /tmp/Llama-2-7b-hf \
    --hf-token <ACCESS TOKEN>

This command will also download the model tokenizer and some other helpful files such as a Responsible Use guide.

|

Fine-tuning Llama3.1-8B-Instruct in torchtune
-------------------------------------------

Torchtune provides `LoRA <https://arxiv.org/abs/2106.09685>`_, `QLoRA <https://arxiv.org/abs/2305.14314>`_, full fine-tuning, Knowledge distillation, direct policy optimization and other
recipes for fine-tuning Llama3-8.1B on one or more GPUs.

Let’s look at how to fine-tune Llama3.1-8B-Instruct using LoRA on a single Intel GPU. In this example, we’ll fine-tune for one epoch on a standard instruction dataset for demonstration.

There are two ways to enable Intel GPU support:

- Pass `device=xpu` directly in command line
 
.. code-block:: bash

    tune run lora_finetune_single_device --config llama3/8B_lora_single_device device=xpu
  
You can also use :ref:`command-line overrides <cli_override>` as needed, e.g.

.. code-block:: bash

    tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device \
        device=xpu \
        checkpointer.checkpoint_dir=<checkpoint_dir> \
        tokenizer.path=<checkpoint_dir>/tokenizer.model \
        checkpointer.output_dir=<checkpoint_dir>

- Manually set `device: xpu`, and other parameters in the configeration file `8B_lora_single_device.yaml`

.. code-block:: yaml

  # Set the correct path to your downloaded tokenizer
  tokenizer:
    path: <checkpoint_dir>/tokenizer.model

  # Set the checkpoint directory and filenames
  checkpointer:
    checkpoint_dir: <checkpoint_dir>
    checkpoint_files: [
      model-00001-of-00004.safetensors,
      model-00002-of-00004.safetensors,
      model-00003-of-00004.safetensors,
      model-00004-of-00004.safetensors
    ]

  # Set device to use Intel GPU
  device: xpu
  dtype: bf16

All other values in the default config can remain unchanged and kick the training.

.. code-block:: bash

    tune run lora_finetune_single_device --config llama3/8B_lora_single_device

.. note::
    To see a full list of recipes and their corresponding configs, simply run ``tune ls`` from the command line.


This will load the Llama3.1-8B-Instruct checkpoint and tokenizer from ``<checkpoint_dir>`` used in the :ref:`tune download <tune_download_label>` command above,
then save a final checkpoint in the same directory following the original format. For more details on the
checkpoint formats supported in torchtune, see our :ref:`checkpointing deep-dive <understand_checkpointer>`.

Once running, you should see logs indicating successful initialization and the start of training:

.. code-block:: bash

  $ tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device device=xpu epochs=1
  INFO:torchtune.utils._logging:Tokenizer is initialized from file.
  INFO:torchtune.utils._logging:Optimizer and loss are initialized.
  INFO:torchtune.utils._logging:Loss is initialized.
  INFO:torchtune.utils._logging:Learning rate scheduler is initialized.
  WARNING:torchtune.utils._logging: Profiling disabled.
  INFO:torchtune.utils._logging: Profiler config after instantiation: {'enabled': False}
  Writing logs to /tmp/torchtune/llama3_1_8B/lora_single_device/logs/log_1744238744.txt
   0%|          | 0/3235 [00:00<?, ?it/s]^M  0%|          | 1/3235 [00:05<4:41:49,  5.23s/it]


You can monitor the loss and progress through the `tqdm <https://tqdm.github.io/>`_ bar but torchtune
will also log some more metrics, such as GPU memory usage, at an interval defined in the config.

|

Once training is complete, the model checkpoints will be saved, and their locations logged. For LoRA fine-tuning, the final checkpoint includes: 

- Merged model weights
- A smaller file with only the LoRA weights

If you want to reduce memory usage further, you can use the QLoRA recipe:

.. code-block:: bash

    tune run lora_finetune_single_device --config llama3_1/8B_qlora_single_device device=xpu

|

Next steps
----------

Now that you have trained your first model and set up your environment, checkout 
Now that you’ve set up your environment and trained your first model, check out  
the `torchtune tutorials <https://github.com/pytorch/torchtune/tree/main/docs/source/tutorials>`_ for more recipes. To enable Intel GPU, simply pass `device=xpu`.

.. TODO(Songhappy) Dstributed finetune a model on Intl GPU
If you have multiple GPUs available, you can run the distributed version of the recipe. 
torchtune makes use of the `FSDP <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`_ APIs from PyTorch Distributed
to shard the model, optimizer states, and gradients. This should enable you to increase your batch size, resulting in faster overall training.
For example, on two devices:

.. code-block:: bash

    tune run --nproc_per_node 2 lora_finetune_distributed --config llama3_1/8B_lora device=xpu


