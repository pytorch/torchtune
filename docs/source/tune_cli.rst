.. _cli_label:

=============
torchtune CLI
=============

This page is the documentation for using the torchtune CLI - a convenient way to
download models, find and copy relevant recipes/configs, and run recipes. It is automatically
available when you install torchtune.

Getting started
---------------

The ``--help`` option will show all the possible commands available through the torchtune CLI,
with a short description of each.

.. code-block:: bash

    $ tune --help
    usage: tune [-h] {download,ls,cp,run,validate,cat} ...

    Welcome to the torchtune CLI!

    options:
    -h, --help            show this help message and exit

    subcommands:
      {download,ls,cp,run,validate,cat}
        download            Download a model from the Hugging Face Hub.
        ls                  List all built-in recipes and configs
        ...

The ``--help`` option is convenient for getting more details about any command. You can use it anytime to list all
available options and their details. For example, ``tune download --help`` provides more information on how
to download files using the CLI.

.. _tune_download_label:

Download a model
----------------

The ``tune download <path>`` command downloads any model from the Hugging Face or Kaggle Model Hub.

.. list-table::
   :widths: 30 60

   * - \--output-dir
     - Directory in which to save the model. Note: this is option not yet supported when `--source` is set to `kaggle`.
   * - \--output-dir-use-symlinks
     - To be used with `output-dir`. If set to 'auto', the cache directory will be used and the file will be either duplicated or symlinked to the local directory depending on its size. It set to `True`, a symlink will be created, no matter the file size. If set to `False`, the file will either be duplicated from cache (if already exists) or downloaded from the Hub and not cached.
   * - \--hf-token
     - Hugging Face API token. Needed for gated models like Llama.
   * - \--ignore-patterns
     - If provided, files matching any of the patterns are not downloaded. Defaults to ignoring safetensors files to avoid downloading duplicate weights.
   * - \--source {huggingface,kaggle}
     - If provided, downloads model weights from the provided <path> on the designated source hub.
   * - \--kaggle-username
     - Kaggle username for authentication. Needed for private models or gated models like Llama2.
   * - \--kaggle-api-key
     - Kaggle API key. Needed for private models or gated models like Llama2. You can find your API key at https://kaggle.com/settings.

.. code-block:: bash

    $ tune download meta-llama/Meta-Llama-3-8B-Instruct
    Successfully downloaded model repo and wrote to the following locations:
    ./model/config.json
    ./model/README.md
    ./model/model-00001-of-00002.bin
    ...

.. code-block:: bash

    $ tune download metaresearch/llama-3.2/pytorch/1b --source kaggle
    Successfully downloaded model repo and wrote to the following locations:
    /tmp/llama-3.2/pytorch/1b/tokenizer.model
    /tmp/llama-3.2/pytorch/1b/params.json
    /tmp/llama-3.2/pytorch/1b/consolidated.00.pth

**Download a gated model**

A lot of recent large pretrained models released from organizations like Meta or MistralAI require you to agree
to the usage terms and conditions before you are allowed to download their model. If this is the case, you can specify
a Hugging Face access token.

You can find the access token `here <https://huggingface.co/docs/hub/en/security-tokens>`_.

.. code-block:: bash

    $ tune download meta-llama/Meta-Llama-3-8B-Instruct --hf-token <TOKEN>
    Successfully downloaded model repo and wrote to the following locations:
    ./model/config.json
    ./model/README.md
    ./model/model-00001-of-00002.bin
    ...

.. note::
    If you'd prefer, you can also use ``huggingface-cli login`` to permanently login to the Hugging Face Hub on your machine.
    The ``tune download`` command will pull the access token from your environment.

**Specify model files you don't want to download**

Some checkpoint directories can be very large and it can eat up a lot of bandwith and local storage to download the all of the files every time, even if you might
not need a lot of them. This is especially common when the same checkpoint exists in different formats. You can specify patterns to ignore to prevent downloading files
with matching names. By default we ignore safetensor files, but if you want to include all files you can pass in an empty string.

.. code-block:: bash

    $ tune download meta-llama/Meta-Llama-3-8B-Instruct --hf-token <TOKEN> --ignore-patterns None
    Successfully downloaded model repo and wrote to the following locations:
    ./model/config.json
    ./model/README.md
    ./model/model-00001-of-00030.safetensors
    ...

.. note::
    Just because a model can be downloaded does not mean that it will work OOTB with torchtune's
    built-in recipes or configs. For a list of supported model families and architectures, see :ref:`models<models>`.


.. _tune_ls_label:

List built-in recipes and configs
---------------------------------

The ``tune ls`` command lists out all the built-in recipes and configs within torchtune. Include the `--experimental` flag to view experimental new recipes as well.


.. code-block:: bash

    $ tune ls
    RECIPE                                   CONFIG
    full_finetune_single_device              llama2/7B_full_low_memory
                                             llama3/8B_full_single_device
                                             mistral/7B_full_low_memory
                                             phi3/mini_full_low_memory
    full_finetune_distributed                llama2/7B_full
                                             llama2/13B_full
                                             llama3/8B_full
                                             llama3/70B_full
    ...

.. _tune_cp_cli_label:

Copy a built-in recipe or config
--------------------------------

The ``tune cp <recipe|config> <path>`` command copies built-in recipes and configs to a provided location. This allows you to make a local copy of a library
recipe or config to edit directly for yourself. See :ref:`here <tune_cp_label>` for an example of how to use this command.

.. list-table::
   :widths: 30 60

   * - \-n, \--no-clobber
     - Do not overwrite destination if it already exists
   * - \--make-parents
     - Create parent directories for destination if they do not exist. If not set to True, will error if parent directories do not exist

.. code-block:: bash

    $ tune cp lora_finetune_distributed .
    Copied file to ./lora_finetune_distributed.py

Run a recipe
------------

The ``tune run <recipe> --config <config>`` is a wrapper around `torchrun <https://pytorch.org/docs/stable/elastic/run.html>`_. ``tune run`` allows you to specify
a built-in recipe or config by name, or by path to use your local recipes/configs.

To run a tune recipe

.. code-block:: bash

    tune run lora_finetune_single_device --config llama3/8B_lora_single_device

**Specifying distributed (torchrun) arguments**

``tune run`` supports launching distributed runs by passing through arguments preceding the recipe directly to torchrun. This follows the pattern used by torchrun
of specifying distributed and host machine flags before the script (recipe). For a full list of available flags for distributed setup, see the `torchrun docs <https://pytorch.org/docs/stable/elastic/run.html>`_.

Some common flags:

.. list-table::
   :widths: 30 60

   * - \--nproc-per-node
     - Number of workers per node; supported values: [auto, cpu, gpu, int].
   * - \--nnodes
     - Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.
   * - \--max-restarts
     - Maximum number of worker group restarts before failing.
   * - \--rdzv-backend
     - Rendezvous backend.
   * - \--rdzv-endpoint
     - Rendezvous backend endpoint; usually in form <host>:<port>.

.. code-block:: bash

    tune run --nnodes=1 --nproc-per-node=4 lora_finetune_distributed --config llama3/8B_lora

.. note::
    If no arguments are provided before the recipe, tune will bypass torchrun and launch directly with ``python``. This can simplify running and debugging recipes
    when distributed isn't needed. If you want to launch with torchrun, but use only a single device, you can specify ``tune run --nnodes=1 --nproc-per-node=1 <recipe> --config <config>``.

**Running a custom (local) recipe and config**

To use ``tune run`` with your own local recipes and configs, simply pass in a file path instead of a name to the run command. You can mix and match a custom recipe with a
torchtune config or vice versa or you can use both custom configs and recipes.

.. code-block:: bash

    tune run my/fancy_lora.py --config my/configs/8B_fancy_lora.yaml

**Overriding the config**

You can override existing parameters from the command line using a key=value format. Let’s say you want to set the number of training epochs to 1.
Further information on config overrides can be found :ref:`here  <cli_override>`.

.. code-block:: bash

  tune run <RECIPE> --config <CONFIG> epochs=1

.. _validate_cli_label:

Validate a config
-----------------

The ``tune validate <config>`` command will validate that your config is formatted properly.


.. code-block:: bash

    # If you've copied over a built-in config and want to validate custom changes
    $ tune validate my_configs/llama3/8B_full.yaml
    Config is well-formed!

.. _tune_cat_cli_label:

Inspect a config
---------------------

The ``tune cat <config>`` command pretty prints a configuration file, making it easy to use ``tune run`` with confidence. This command is useful for inspecting the structure and contents of a config file before running a recipe, ensuring that all parameters are correctly set.

You can also use the ``--sort`` option to print the config in sorted order, which can help in quickly locating specific keys.

.. list-table::
   :widths: 30 60

   * - \--sort
     - Print the config in sorted order.

**Workflow Example**

1. **List all available configs:**

   Use the ``tune ls`` command to list all the built-in recipes and configs within torchtune.

   .. code-block:: bash

       $ tune ls
       RECIPE                                   CONFIG
       full_finetune_single_device              llama2/7B_full_low_memory
                                                llama3/8B_full_single_device
                                                mistral/7B_full_low_memory
                                                phi3/mini_full_low_memory
       full_finetune_distributed                llama2/7B_full
                                                llama2/13B_full
                                                llama3/8B_full
                                                llama3/70B_full
       ...

2. **Inspect the contents of a config:**

   Use the ``tune cat`` command to pretty print the contents of a specific config. This helps you understand the structure and parameters of the config.

   .. code-block:: bash

       $ tune cat llama2/7B_full
       output_dir: /tmp/torchtune/llama2_7B/full
       tokenizer:
           _component_: torchtune.models.llama2.llama2_tokenizer
           path: /tmp/Llama-2-7b-hf/tokenizer.model
           max_seq_len: null
       ...

   You can also print the config in sorted order:

   .. code-block:: bash

       $ tune cat llama2/7B_full --sort

3. **Run a recipe with parameter override:**

   After inspecting the config, you can use the ``tune run`` command to run a recipe with the config. You can also override specific parameters directly from the command line. For example, to override the `output_dir` parameter:

   .. code-block:: bash

       $ tune run full_finetune_distributed --config llama2/7B_full output_dir=./

   Learn more about config overrides :ref:`here  <cli_override>`.

.. note::
    You can find all the cat-able configs via the ``tune ls`` command.
