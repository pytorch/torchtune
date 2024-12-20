.. _finetune_llama_label:

========================
Fine-Tune Your First LLM
========================

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

For this tutorial, you're going to use the `Llama2 7B model from Meta <https://llama.meta.com/>`_. Llama2 is a "gated model",
meaning that you need to be granted access in order to download the weights. Follow `these instructions <https://huggingface.co/meta-llama>`_ on the official Meta page
hosted on Hugging Face to complete this process. This should take less than 5 minutes. To verify that you have the access, go to the `model page <https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main>`_.
You should be able to see the model files. If not, you may need to accept the agreement to complete the process.

.. note::

  Alternatively, you can opt to download the model directly through the Llama2 repository.
  See `this page <https://llama.meta.com/get-started#getting-the-models>`_ for more details.

Once you have authorization, you will need to authenticate with Hugging Face Hub. The easiest way to do so is to provide an
access token to the download script. You can find your token `here <https://huggingface.co/settings/tokens>`_.

Then, it's as simple as:

.. code-block:: bash

  tune download meta-llama/Llama-2-7b-hf \
    --output-dir /tmp/Llama-2-7b-hf \
    --hf-token <ACCESS TOKEN>

This command will also download the model tokenizer and some other helpful files such as a Responsible Use guide.

|

Selecting a recipe
------------------
Recipes are the primary entry points for torchtune users.
These can be thought of as **hackable, singularly-focused scripts for interacting with LLMs** including training,
inference, evaluation, and quantization.

Each recipe consists of three components:

* **Configurable parameters**, specified through yaml configs and command-line overrides
* **Recipe script**, entry-point which puts everything together including parsing and validating configs, setting up the environment, and correctly using the recipe class
* **Recipe class**, core logic needed for training, exposed through a set of APIs

.. note::

  To learn more about the concept of "recipes", check out our technical deep-dive: :ref:`recipe_deepdive`.

torchtune provides built-in recipes for finetuning on single device, on multiple devices with `FSDP <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/>`_,
using memory efficient techniques like `LoRA <https://arxiv.org/abs/2106.09685>`_, and more! Check out all our built-in recipes in our :ref:`recipes overview<recipes_overview_label>`. You can also utilize the
:code:`tune ls` command to print out all recipes and corresponding configs.

.. code-block:: bash

  $ tune ls
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
a single device. For a more in-depth discussion on LoRA in torchtune, you can see the complete ":ref:`lora_finetune_label`" tutorial.

.. note::

  **Why have a separate recipe for single device vs. distributed?** This is discussed in
  ":ref:`recipe_deepdive`" but one of our :ref:`core principles <design_principles_label>` in torchtune is minimal abstraction and boilerplate code.
  If you only want to train on a single GPU, our single-device recipe ensures you don't have to worry about additional
  features like FSDP that are only required for distributed training.

|

.. _tune_cp_label:

Modifying a config
------------------
YAML configs hold most of the important information needed for running your recipe.
You can set hyperparameters, specify metric loggers like `WandB <wandb.ai>`_, select a new dataset, and more.
For a list of all currently supported datasets, see :ref:`datasets`.

There are two ways to modify an existing config:

**Override existing parameters from the command line**

You can override existing parameters from the command line using a :code:`key=value` format. Let's say
you want to set the number of training epochs to 1.

.. code-block:: bash

  tune run <RECIPE> --config <CONFIG> epochs=1

**Copy the config through `tune cp` and modify directly**

If you want to make more substantial changes to the config, you can use the :ref:`tune <cli_label>` CLI to copy it to your local directory.

.. code-block:: bash

  $ tune cp llama2/7B_lora_single_device custom_config.yaml
  Copied file to custom_config.yaml

Now you can update the custom YAML config any way you like. Try setting the random seed in order to make replication easier,
changing the LoRA rank, update batch size, etc.

.. note::

  Check out ":ref:`config_tutorial_label`" for a deeper dive on configs in torchtune.

|

Training a model
----------------
Now that you have a model in the proper format and a config that suits your needs, let's get training!

Just like all the other steps, you will be using the tune CLI tool to launch your finetuning run.

.. code-block:: bash

  $ tune run lora_finetune_single_device --config llama2/7B_lora_single_device epochs=1
  INFO:torchtune.utils.logging:Running LoRAFinetuneRecipeSingleDevice with resolved config:
  Writing logs to /tmp/lora_finetune_output/log_1713194212.txt
  INFO:torchtune.utils.logging:Model is initialized with precision torch.bfloat16.
  INFO:torchtune.utils.logging:Tokenizer is initialized from file.
  INFO:torchtune.utils.logging:Optimizer and loss are initialized.
  INFO:torchtune.utils.logging:Loss is initialized.
  INFO:torchtune.utils.logging:Dataset and Sampler are initialized.
  INFO:torchtune.utils.logging:Learning rate scheduler is initialized.
  1|52|Loss: 2.3697006702423096:   0%|▏                     | 52/25880 [00:24<3:55:01,  1.83it/s]

You can see that all the modules were successfully initialized and the model has started training.
You can monitor the loss and progress through the `tqdm <https://tqdm.github.io/>`_ bar but torchtune
will also log some more metrics, such as GPU memory usage, at an interval defined in the config.

|

Next steps
----------

Now that you have trained your model and set up your environment, let's take a look at what we can do with our
new model by checking out the ":ref:`E2E Workflow Tutorial<e2e_flow>`".
