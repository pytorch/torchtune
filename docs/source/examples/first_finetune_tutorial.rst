===============================
Finetune Llama2 with TorchTune
===============================

This tutorial will guide you through the process of launching your first finetuning
job using TorchTune.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * How to download a model and convert it to a format compatible with Torchtune
      * How to modify a recipe's parameters
      * How to finetune a model

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with the :ref:`overview of TorchTune<overview_label>`
      * Make sure to :ref:`install TorchTune<install_label>`


Downloading a model
-------------------
First, you need to download a model. The first way to download a model is to use TorchTune's integration
with the `HuggingFace Hub <https://huggingface.co/docs/hub/en/index>`_ - a collection of the latest and greatest model weights.

For this tutorial, you're going to use the `Llama2 model from Meta <https://llama.meta.com/>`_. Llama2 is a "gated model",
meaning that you need to be granted access in order to download the weights. Follow `these instructions <https://huggingface.co/meta-llama>`_ on the official Meta page
hosted on HuggingFace to complete this process. (This should take less than 5 minutes.)

Once you have authorization, you will need to authenticate with HuggingFace Hub. The easiest way to do so is to provide an
access token to the download script. You can find your token by visiting https://huggingface.co/settings/tokens.

Then, it's as simple as:

.. code-block:: bash

  tune download \
  --repo-id meta-llama/Llama-2-7b \
  --output-dir /tmp/llama2 \
  --hf-token <ACCESS TOKEN>

This command will also download the model tokenizer and some other helpful files such as a Responsible Use guide.

.. note::

  You can also download the model directly through the Llama2 repository.
  See https://llama.meta.com/get-started#getting-the-models for more details.

Converting model weights
------------------------
TorchTune has modular native-PyTorch implementations of popular LLMs. To convert a model to this format, use the following command:

.. code-block:: bash

  tune convert_checkpoint --checkpoint-path /tmp/llama2/consolidated.00.pth

By default, this will output a file to the same directory as the checkpoint with the name `native_pytorch_model.pt`

Selecting a recipe
------------------
Recipes are the primary entry points for TorchTune users.
These can be thought of as end-to-end pipelines for training and optionally evaluating LLMs.

Each recipe consists of three components:

* **Configurable parameters**, specified through yaml configs, command-line overrides and dataclasses
* **Recipe class**, core logic needed for training, exposed to users through a set of APIs
* **Recipe script**, puts everything together including parsing and validating configs, setting up the environment, and correctly using the recipe class

To see all available recipes and for more information on how to select the right recipe, see :ref:`recipe_deepdive`.
For this tutorial, you'll be using the :ref:`basic full finetuning recipe<basic_finetune_llm>`.

Modifying a config
------------------
YAML configs hold most of the important information needed for running your recipe.
You can select a new dataset, set hyperparameters, specify metric loggers like `WandB <wandb.ai>`_, and more.

.. note::
  For a list of all currently supported datasets, see :ref:`datasets`.

To modify an existing recipe config, you can use the :code:`tune` CLI to copy it to your local directory.
Or, you can visit the specific :ref:`recipe page<basic_finetune_llm>` and copy/paste the config from there.
It looks like there's already a config called :code:`alpaca_llama_full_finetune` that utilizes the popular
`Alpaca instruction dataset <https://crfm.stanford.edu/2023/03/13/alpaca.html>`_. This seems like a good place to start so let's copy it!

.. code-block:: bash

  tune config cp alpaca_llama2_full_finetune custom_config.yaml

Now you can update the custom YAML config to point to your model and tokenizer. While you're at it,
you can make some other changes, like setting the random seed in order to make replication easier,
lowering the epochs to 1 so you can see results sooner, and updating the learning rate.

.. code-block:: yaml

  # Dataset and Dataloader
  dataset: alpaca
  seed: 42
  shuffle: True

  # Model Arguments
  model: llama2_7b
  model_checkpoint: /tmp/llama2/native_pytorch_model.pt
  tokenizer: llama2_tokenizer
  tokenizer_checkpoint: /tmp/llama2/tokenizer.model

  # Fine-tuning arguments
  batch_size: 2
  lr: 1e-5
  epochs: 1
  optimizer: SGD
  loss: CrossEntropyLoss
  output_dir: /tmp/alpaca-llama2-finetune
  device: cuda
  dtype: fp32
  enable_fsdp: True
  enable_activation_checkpointing: True
  resume_from_checkpoint: False


Training a model
----------------
Now that you have a model in the proper format and a config that suits your needs, let's get training!

Just like all the other steps, you will be using the :code:`tune` CLI tool to launch your finetuning run.
To make it easier for users already familiar with the PyTorch ecosystem, TorchTune integrates with
`torchrun <https://pytorch.org/docs/stable/elastic/run.html>`_. Therefore, in order to launch a distributed
run using two GPUs, it's as easy as:

.. code-block:: bash

  tune --nnodes 1 --nproc_per_node 2 full_finetune.py --config custom_config.yaml

You should see some immediate output and see the loss going down, indicating your model is training!

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

There's a lot more you can accomplish using TorchTune, including:

* Using your finetuned model to generate some output
* Evaluating your finetuned model on common benchmarks using `Eluther AI Eval Harness <https://www.eleuther.ai/projects/large-language-model-evaluation>`_
* Outputting metrics to `WandB <wandb.ai>`_ or `Tensorboard <https://www.tensorflow.org/tensorboard/>`_
