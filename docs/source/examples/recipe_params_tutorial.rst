======================================
Creating parameters for custom recipes
======================================

In general, you should expose the minimal amount of parameters you need to run and experiment with your recipes. These should be collected in a dataclass object that is passed into the recipe.

.. code-block:: python

    class FullFinetuneParams:
        # Model
        model: str = ""
        model_checkpoint: str = ""

In the dataclass, all fields should have defaults assigned to them. If a reasonable value cannot be assigned or it is a required argument, use the null value for that data type as the default and ensure that it is set by the user in the :code:`__post_init__` (see Parameter Validation). The dataclass should go in the :code:`recipes/params/` folder and the name of the file should match the name of the recipe file you are creating.

To link the dataclass object with config and CLI parsing, you can use the :class:`~torchtune.utils.argparse.TuneArgumentParser` object and funnel the parsed arguments into your dataclass.

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

Parameter validation
--------------------
To validate user arguments for your dataclass and recipe, use the :code:`__post_init__` method to house any checks and raised exceptions.

.. code-block:: python

    def __post_init__(self):
        for param in fields(self):
            if getattr(self, param.name) == "":
                raise TypeError(f"{param.name} needs to be specified")

Write config
------------
Now that you've set up the recipe, the parameters dataclass, and the parser, you can create a simple config in :code:`recipes/configs/` that specifies values for any of the fields you defined in the dataclass. Anything that is not specified should have a default value in the dataclass.

Testing configs
---------------
TorchTune has testing for every config added to the library, namely ensuring that it instantiates the dataclass and runs the recipe correctly. To add your config to this test suite, simply update the dictionary in :code:`recipes/tests/configs/test_configs.py`.

.. code-block:: python

    config_to_params = {
        os.path.join(ROOT_DIR, "alpaca_llama2_full_finetune.yaml"): FullFinetuneParams,
        ...,
    }


Running your recipe
-------------------
If everything is set up correctly, you should be able to run your recipe just like the existing library recipes using the :code:`tune` command:

.. code-block:: bash

    tune <recipe> --config <config> --override ...
