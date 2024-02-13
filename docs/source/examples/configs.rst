.. _config_tutorial_label:

=================
Configs Deep-Dive
=================

This tutorial will guide you through writing configs for running recipes and
designing params for custom recipes.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * How to write a YAML config and run a recipe with it
      * How to create a params dataclass for custom recipe
      * How to effectively use configs, CLI overrides, and dataclasses for running recipes

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with the :ref:`overview of TorchTune<overview_label>`
      * Make sure to :ref:`install TorchTune<install_label>`
      * Understand the :ref:`fundamentals of recipes<recipe_deepdive>`


Where do parameters live?
-------------------------

There are two primary entry points for you to configure parameters: **configs** and
**CLI overrides**. Configs are YAML files that define all the
parameters needed to run a recipe within a single location. These can be overridden on the
command-line for quick changes and experimentation without modifying the config.

If you are planning to make a custom recipe, you will need to become familiar
with the **recipe dataclass**, which collects all of your arguments from config and
CLI, and passes it into the recipe itself. Here, we will discuss all three concepts:
**configs**, **CLI**, and **dataclasses**.


Recipe dataclasses
------------------

Parameters should be organized in a single dataclass that is passed into the recipe.
This serves as a single source of truth for the details of a fine-tuning run that can be easily validated in code and shared with collaborators for reproducibility.

.. code-block:: python

    class FullFinetuneParams:
        # Model
        model: str = ""
        model_checkpoint: str = ""

In the dataclass, all fields should have defaults assigned to them.
If a reasonable value cannot be assigned or it is a required argument,
use the null value for that data type as the default and ensure that it is set
by the user in the :code:`__post_init__` (see :ref:`Parameter Validation<parameter_validation_label>`).
The dataclass should go in the :code:`recipes/params/` folder and the name of
the file should match the name of the recipe file you are creating.

In general, you should expose the minimal amount of parameters you need to run and experiment with your recipes.
Exposing an excessive number of parameters will lead to bloated configs, which are more error prone, harder to read, and harder to manage.
On the other hand, hardcoding all parameters will prevent quick experimentation without a code change. Only parametrize what is needed.

To link the dataclass object with config and CLI parsing,
you can use the :class:`~torchtune.utils.argparse.TuneArgumentParser` object and
funnel the parsed arguments into your dataclass.

.. code-block:: python

    if __name__ == "__main__":
        parser = utils.TuneArgumentParser(
            description=FullFinetuneParams.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        # Get user-specified args from config and CLI and create params for recipe
        args, _ = parser.parse_known_args()
        args = vars(args)
        params = FullFinetuneParams(**args)

        logger = utils.get_logger("DEBUG")
        logger.info(msg=f"Running finetune_llm.py with parameters {params}")

        recipe(params)

.. _parameter_validation_label:

Parameter validation
--------------------
To validate arguments for your dataclass and recipe, use the :code:`__post_init__` method to house any checks and raised exceptions.

.. code-block:: python

    def __post_init__(self):
        for param in fields(self):
            if getattr(self, param.name) == "":
                raise TypeError(f"{param.name} needs to be specified")

Writing configs
---------------
Once you've set up a recipe and its params, you need to create a config to run it.
Configs serve as the primary entry point for running recipes in TorchTune. They are
expected to be YAML files and simply list out values for parameters you want to define
for a particular run. The config parameters should be a subset of the dataclass parameters;
there should not be any new fields that are not already in the dataclass. Any parameters that
are not specified in the config will take on the default value defined in the dataclass.

.. code-block:: yaml

    dataset: alpaca
    seed: null
    shuffle: True
    model: llama2_7b
    ...

Command-line overrides
----------------------
To enable quick experimentation, you can specify override values to parameters in your config
via the :code:`tune` command. These should be specified with the flag :code:`--override k1=v1 k2=v2 ...`

For example, to run the :code:`full_finetune` recipe with custom model and tokenizer directories and using GPUs, you can provide overrides:

.. code-block:: bash

    tune full_finetune --config alpaca_llama2_full_finetune --override model_directory=/home/my_model_checkpoint tokenizer_directory=/home/my_tokenizer_checkpoint device=cuda

The order of overrides from these parameter sources is as follows, with highest precedence first: CLI, Config, Dataclass defaults


Testing configs
---------------
If you plan on contributing your config to the repo, we recommend adding it to the testing suite. TorchTune has testing for every config added to the library, namely ensuring that it instantiates the dataclass and runs the recipe correctly.

To add your config to this test suite, simply update the dictionary in :code:`recipes/tests/configs/test_configs`.

.. code-block:: python

    config_to_params = {
        os.path.join(ROOT_DIR, "alpaca_llama2_full_finetune.yaml"): FullFinetuneParams,
        ...,
    }

Linking recipes and configs with :code:`tune`
---------------------------------------------

In order to run your custom recipe and configs with :code:`tune`, you must update the :code:`_RECIPE_LIST`
and :code:`_CONFIG_LISTS` in :code:`recipes/__init__.py`

.. code-block:: python

    _RECIPE_LIST = ["full_finetune", "lora_finetune", "alpaca_generate", ...]
    _CONFIG_LISTS = {
        "full_finetune": ["alpaca_llama2_full_finetune"],
        "lora_finetune": ["alpaca_llama2_lora_finetune"],
        "alpaca_generate": [],
        "<your_recipe>": ["<your_config"],
    }

Running your recipe
-------------------
If everything is set up correctly, you should be able to run your recipe just like the existing library recipes using the :code:`tune` command:

.. code-block:: bash

    tune <recipe> --config <config> --override ...
