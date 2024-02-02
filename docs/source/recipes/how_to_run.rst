==================
How to Run Recipes
==================

.. note::
    Ensure that the documentation version matches the installed TorchTune version

The arguments for a recipe are defined in a params object (such as :class:`~recipes.params.FullFinetuneParams`) that contains the full list of configurable parameters. These are either set to default values or sourced from the YAML file listed with :code:`--config`` and :code:`--override` arguments in the tune CLI. The :class:`~torchtune.utils.argparse.TuneArgumentParser` class is responsible for parsing the provided config file and overrides and funneling it into the corresponding params object for the recipe the user wishes to run. The order of overrides from these parameter sources is as follows, with highest precedence first: CLI, Config, Params defaults

The config is the primary entry point for users, with CLI overrides providing flexibility for quick experimentation.

Examples
--------

If you want to run the :code:`finetune_llm` recipe using the :code:`alpaca_llama2_finetune.yaml` config only on CPU, you can provide the config file and override the :code:`device` field.

.. code-block:: bash

    tune finetune_llm --config alpaca_llama2_finetune --override device=cpu
