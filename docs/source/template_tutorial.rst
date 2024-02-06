==============================
Finetune a LLM with TorchTune!
==============================

This tutorial will guide you through the process of using TorchTune to
download a model, finetune it on a dataset, and evaluate its performance.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * How to download a model and convert it to a format compatible with Torchtune
      * How to select a dataset and other parameters
      * How to finetune a model and evaluate its performance

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with the overview of TorchTune [Link Overview]
      * Install TorchTune following the instructions here


Downloading a model
------------------
First, we need to download a model. Let's use the `Llama2 model from Meta__`. The preferred
way to download a model is to use the []HuggingFace Hub. For gated models, like Llama2, you
will need to provide a Hugging Face API token. You can find your token by visiting https://huggingface.co/settings/tokens.
Then, it's as simple as:

.. code-block:: bash

  tune download --repo_id meta-llama/Llama-2-7b --output_dir /tmp/llama2

This will also download the tokenizer and a Responsible Use guide.

Note:
  If you prefer not to download via the HuggingFace Hub, you can also download the model
  directly from the Llama2 GitHub repo.

Converting model weights
------------------------
TorchTune utilizes a PyTorch-native model format. Why? With everything written in PyTorch,
it's easier to debug and modify and allows great interopability with tools like []FSDP.
To convert a model to this format, we use the following command:

.. code-block:: bash

  tune convert_checkpoint --checkpoint_path /tmp/llama2/consolidated.00.pth

By default this will output a file to the same directory as the checkpoint with the name `native_pytorch_model.pt`


Recipes
-------
Recipes are the primary entry points for TorchTune users.
These can be thought of as end-to-end pipelines for training and optionally evaluating LLMs.

Each recipe consists of three components:

* Configurable parameters, specified through yaml configs []example, command-line overrides and dataclasses
* Recipe class, core logic needed for training, exposed to users through a set of APIs []interface
* Recipe script, puts everything together including parsing and validating configs, setting up the environment, and correctly using the recipe class

To see all available recipes, you can visit []XYZ and for more information about how recipes work, see []XYZ.


Selecting a dataset and other parameters
----------------------------------------
Now that we have a model in the correct format, we can get onto more interesting things!
TorchTune integrates with []HuggingFace datasets to make it easy to get started.
To see all supported datasets, you can visit []XYZ. Let's utilize the popular []Alpaca dataset.

To modify an existing recipe config, you can use the `tune` CLI to copy it to your local directory.
Additionally, you can visit the []recipe page and copy the config from there.

.. code-block:: bash

  tune config cp alpaca_llama2_finetune custom_alpaca_llama2_finetune

Training a model
----------------
Finally, we see that
