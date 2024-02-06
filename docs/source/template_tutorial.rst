==============================
Finetune a LLM with TorchTune!
==============================

This tutorial will guide you through the process of using TorchTune to
download a model, select an existing recipe to

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * How to download a model and convert it to a format compatible with Torchtune
      * How to select a dataset and other parameters
      * How to finetune a model and evaluate its performance

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with the :ref:`overview of TorchTune<overview_label>`
      * :ref:`Install TorchTune<install_label>`


Downloading a model
-------------------
First, we need to download a model. The preferred way to download a model is to use the `HuggingFace Hub <https://huggingface.co/docs/hub/en/index>`_
- a collection of the latest and greatest model weights.

For this tutorial, we're going to use the `Llama2 model from Meta <https://llama.meta.com/>`_. Llama2 is a "gated model",
meaning that you need to be granted access in order to access the weights. Follow `these instructions <https://huggingface.co/meta-llama>`_ on the official Meta page
hosted on HuggingFace to complete this process.

Once you have authorization, you will need to authenticate with HuggingFace Hub. The easiest way to do so is to provide an
access token to the download script. You can find your token by visiting https://huggingface.co/settings/tokens.

Then, it's as simple as:

.. code-block:: bash

  tune download --repo_id meta-llama/Llama-2-7b --hf-token <ACCESS TOKEN> --output_dir /tmp/llama2

This command will also download the model tokenizer and a Responsible Use guide.

.. note::

  If you prefer not to download via the HuggingFace Hub, you can also download the model
  directly through the Llama2 repository. See https://llama.meta.com/get-started#getting-the-models
  for more details.

Converting model weights
------------------------
TorchTune utilizes a PyTorch-native model format. Why? With everything written in PyTorch,
it's easier to debug and modify and allows great interopability with tools like `FSDP <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/>`_.
To convert a model to this format, use the following command:

.. code-block:: bash

  tune convert_checkpoint --checkpoint_path /tmp/llama2/consolidated.00.pth

By default this will output a file to the same directory as the checkpoint with the name `native_pytorch_model.pt`

Selecting a recipe
------------------
Recipes are the primary entry points for TorchTune users.
These can be thought of as end-to-end pipelines for training and optionally evaluating LLMs.

Each recipe consists of three components:

* **Configurable parameters**, specified through yaml configs, command-line overrides and dataclasses
* **Recipe class**, core logic needed for training, exposed to users through a set of APIs
* **Recipe script**, puts everything together including parsing and validating configs, setting up the environment, and correctly using the recipe class

Too see all available recipes and for information on how to select the right recipe, see our ``:ref:`recipe tutorial<>``.
For this tutorial, we'll be using the :ref:`basic full finetuning recipe<basic_finetune_llm>`.

Modifying a config
------------------
YAML configs hold most of the important information needed for running your recipe.
You can select a new dataset, set hyperparameters, specify metric loggers like `WandB <wandb.ai>`_, and more!

.. note::
  TorchTune integrates with `HuggingFace datasets <https://huggingface.co/docs/datasets/en/index>`_ to provide
  easy access to the best datasets. For a list of all currently supported datasets, see `:ref:<datasets>`.

To modify an existing recipe config, you can use the `tune` CLI to copy it to your local directory.
Additionally, you can visit the specific :ref:`recipe page<basic_finetune_llm>` and copy/paste the config from there.
It looks like there's already a config called :code:`alpaca_llama_full_finetune` that utilizes the popular
`Alpaca dataset <https://crfm.stanford.edu/2023/03/13/alpaca.html>`_. This seems like a good place to start so let's copy it!

.. code-block:: bash

  tune config cp alpaca_llama2_full_finetune custom_alpaca_llama2_full_finetune

Now we can update the YAML config to point to our model. While we're at it,
we can make some other changes, like setting the random seed in order to make replication easier,
lowering the epochs to 1 so we can see results sooner, and changing the :code:`dtype` to run in half precision.

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
  lr: 2e-5
  epochs: 2
  optimizer: SGD
  loss: CrossEntropyLoss
  output_dir: /tmp/alpaca-llama2-finetune
  device: cuda
  dtype: fp16
  enable_fsdp: True
  enable_activation_checkpointing: True
  resume_from_checkpoint: False


Training a model
----------------
Now that we have a model in the proper format and a config that suits our needs, let's get training!

Just like all the other steps, we will be using the :code:`tune` CLI tool to launch our finetuning run.
To make it easier for users already familiar with the PyTorch ecosystem, TorchTune integrates with
`torchrun <https://pytorch.org/docs/stable/elastic/run.html>`_. Therefore, in order to launch a distributed
run using two GPUs, it's as easy as:

.. code-block:: bash

  tune --nnodes 1 --nproc_per_node 2 full_finetune.py --config <PATH_TO_OUR_CUSTOM_CONFIG>

You should see some immediate output and see the loss going down, indicating your model is training!

`INSERT PICTURE HERE`

Next steps
----------

There's a lot more you can accomplish using TorchTune, including (but not limited to):
* Using your finetuned model to generate some output
* Evaluating your finetuned model on common benchmarks using Eluther ai
* Outputting metrics to WandB or Tensorboard
