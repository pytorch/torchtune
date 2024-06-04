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
    usage: tune [-h] {download,ls,cp,run,validate} ...

    Welcome to the torchtune CLI!

    options:
    -h, --help            show this help message and exit

    subcommands:
      {download,ls,cp,run,validate}
        download            Download a model from the Hugging Face Hub.
        ls                  List all built-in recipes and configs
        ...

The ``--help`` option is convenient for getting more details about any command. You can use it anytime to list all
available options and their details. For example, ``tune download --help`` provides more information on how
to download files using the CLI.

Download a model
----------------

The ``tune download`` command downloads any model from the Hugging Face Hub.

.. code-block:: bash
    $ tune download meta-llama/Meta-Llama-3-8B-Instruct
    Successfully downloaded model repo and wrote to the following locations:
    ./model/config.json
    ./model/README.md
    ./model/model-00001-of-00002.bin
    ...

Download a gated model
^^^^^^^^^^^^^^^^^^^^^^

A lot of recent large pretrained models released from organizations like Meta or MistralAI require you to agree
to the usage terms and conditions before you are allowed to download their model. If this is the case, you can specify
a Hugging Face access token.

You can find the access token here.

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

Specify model files you don't want to download
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. note::
    Just because a model can be downloaded does not mean that it will work OOTB with torchtune's
    built-in recipes or configs. For a list of supported model families and architectures, see <link to models page>.


List built-in recipes and configs
---------------------------------

The ``ls`` command lists out all the built-in recipes and configs within torchtune.

.. code-block:: bash


Copy a built-in recipe or config
--------------------------------

Run a recipe
------------

Specifying distributed arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running a custom (local) recipe
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running a built-in recipe with a custom (local) config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
