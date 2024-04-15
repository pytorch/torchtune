.. _finetune_llama_label:

=======================
Finetune your first LLM
=======================

This guide will walk you through the process of launching your first finetuning
job using torchtune.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * How to download a model from the `Hugging Face Hub <https://huggingface.co/docs/hub/en/index>`_
      * How to modify a recipe's parameters to suit your needs
      * How to run a finetune

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with the :ref:`overview of torchtune<overview_label>`
      * Make sure to :ref:`install torchtune<install_label>`

.. _download_llama_label:

Downloading a model
-------------------
The first step in any finetuning job is to download a pretrained base model. torchtune supports an integration
with the `Hugging Face Hub <https://huggingface.co/docs/hub/en/index>`_ - a collection of the latest and greatest model weights.

For this tutorial, you're going to use the `Llama2 model from Meta <https://llama.meta.com/>`_. Llama2 is a "gated model",
meaning that you need to be granted access in order to download the weights. Follow `these instructions <https://huggingface.co/meta-llama>`_ on the official Meta page
hosted on Hugging Face to complete this process. This should take less than 5 minutes. To verify that you have the access, go to the `model page <https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main>`_.
You should be able to see the model files. If not, you may need to accept the agreement to complete the process.

Once you have authorization, you will need to authenticate with Hugging Face Hub. The easiest way to do so is to provide an
access token to the download script. You can find your token `here <https://huggingface.co/settings/tokens>`_.

Then, it's as simple as:

.. code-block:: bash

  tune download meta-llama/Llama-2-7b-hf \
    --output-dir /tmp/Llama-2-7b-hf \
    --hf-token <ACCESS TOKEN>

This command will also download the model tokenizer and some other helpful files such as a Responsible Use guide.

.. note::

  You can opt to download the model directly through the Llama2 repository.
  See `this page <https://llama.meta.com/get-started#getting-the-models>`_ for more details.


Selecting a recipe
------------------
Recipes are the primary entry points for torchtune users.
These can be thought of as singularly-focused scripts for interacting with LLMs, including training
inference, evaluation, and quantization.

Each recipe consists of three components:

* **Configurable parameters**, specified through yaml configs and command-line overrides
* **Recipe script**, entry-point which puts everything together including parsing and validating configs, setting up the environment, and correctly using the recipe class
* **Recipe class**, core logic needed for training, exposed to users through a set of APIs

.. note::

  To learn more about the concept of "recipes", check out our technical deep-dive: :ref:`recipe_deepdive`.

torchtune provides built-in recipes for finetuning on single device, on multiple devices with `FSDP <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/>`_,
using memory efficient techniques like LoRA, and more! You can view all built-in recipes `here <https://github.com/pytorch/torchtune/tree/main/recipes>`_. You can also utilize the
:code:`tune ls` command to print out all recipes and corresponding configs.

.. code-block:: bash

  tune ls

.. code-block:: text

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

For the purposes of this tutorial, you'll will be using the recipe for finetuning a Llama2 model using `LoRA <https://arxiv.org/abs/2106.09685>`_ on
a single device. For a more in-depth discussion on LoRA in torchtune, you can see the complete :ref:`lora_finetune_label` tutorial.

Modifying a config
------------------
YAML configs hold most of the important information needed for running your recipe.
You can set hyperparameters, specify metric loggers like `WandB <wandb.ai>`_, select a new dataset, and more.
For a list of all currently supported datasets, see :ref:`datasets`.

To modify an existing recipe config, you can use the :code:`tune` CLI to copy it to your local directory.
It looks like there's already a config called :code:`llama2/7B_lora_single_device` that utilizes the popular
`Alpaca instruction dataset <https://crfm.stanford.edu/2023/03/13/alpaca.html>`_. This seems like a good place to start so let's copy it!

.. code-block:: bash

  tune cp llama2/7B_lora_single_device custom_config.yaml

.. code-block:: text

  Copied file to custom_config.yaml

Now you can update the custom YAML config to point to your model and tokenizer (not needed in this case). While you're at it,
you can make some other changes, like setting the random seed in order to make replication easier,
lowering the epochs to 1 so you can see results sooner, and updating the learning rate.

.. code-block:: yaml

  # Model Arguments
  model:
    _component_: torchtune.models.llama2.lora_llama2_7b
    lora_attn_modules: ['q_proj', 'v_proj']
    apply_lora_to_mlp: False
    apply_lora_to_output: False
    lora_rank: 8
    lora_alpha: 16

  tokenizer:
    _component_: torchtune.models.llama2.llama2_tokenizer
    path: /tmp/Llama-2-7b-hf/tokenizer.model

  checkpointer:
    _component_: torchtune.utils.FullModelHFCheckpointer
    checkpoint_dir: /tmp/Llama-2-7b-hf
    checkpoint_files: [
      pytorch_model-00001-of-00002.bin,
      pytorch_model-00002-of-00002.bin
    ]
    adapter_checkpoint: null
    recipe_checkpoint: null
    output_dir: /tmp/Llama-2-7b-hf
    model_type: LLAMA2
  resume_from_checkpoint: False

  # Dataset and Sampler
  dataset:
    _component_: torchtune.datasets.alpaca_cleaned_dataset
    train_on_input: True
  seed: null
  shuffle: True
  batch_size: 2

  # Optimizer and Scheduler
  optimizer:
    _component_: torch.optim.AdamW
    weight_decay: 0.01
    lr: 3e-4
  lr_scheduler:
    _component_: torchtune.modules.get_cosine_schedule_with_warmup
    num_warmup_steps: 100

  loss:
    _component_: torch.nn.CrossEntropyLoss

  # Training
  epochs: 1
  max_steps_per_epoch: null
  gradient_accumulation_steps: 64
  compile: False

  # Logging
  output_dir: /tmp/lora_finetune_output
  metric_logger:
    _component_: torchtune.utils.metric_logging.DiskLogger
    log_dir: ${output_dir}
  log_every_n_steps: null

  # Environment
  device: cuda
  dtype: bf16
  enable_activation_checkpointing: True

  # Show case the usage of pytorch profiler
  # Set enabled to False as it's only needed for debugging training
  profiler:
    _component_: torchtune.utils.profiler
    enabled: False
    output_dir: /tmp/alpaca-llama2-finetune/torchtune_perf_tracing.json


Training a model
----------------
Now that you have a model in the proper format and a config that suits your needs, let's get training!

Just like all the other steps, you will be using the :code:`tune` CLI tool to launch your finetuning run.
To make it easier for users already familiar with the PyTorch ecosystem, torchtune integrates with
`torchrun <https://pytorch.org/docs/stable/elastic/run.html>`_. Therefore, in order to launch a distributed
run using two GPUs, it's as easy as:

.. code-block:: bash

  tune run --nnodes 1 --nproc_per_node 2 full_finetune_distributed --config custom_config.yaml

You should see some immediate output and see the loss going down, indicating your model is training succesfully.

.. code-block:: text

  Writing logs to /tmp/alpaca-llama2-finetune/log_1707246452.txt
  Setting manual seed to local seed 42. Local seed is seed + rank = 42 + 0
  Model is initialized. FSDP and Activation Checkpointing are enabled.
  Tokenizer is initialized from file.
  Optimizer is initialized.
  Loss is initialized.
  Dataset and Sampler are initialized.
  1|1|Loss: 1.7553404569625854:   0%|                       | 0/13000 [00:03<?, ?it/s]

Next steps
----------

Now that you have trained your model and set up your environment, let's take a look at what we can do with our
new model by checking out the :ref:`E2E Workflow Tutorial<e2e_flow>`.
