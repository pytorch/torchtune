.. _recipe_deepdive:

=========================
Training Recipe Deep-Dive
=========================

This tutorial will walk you through the design of training-recipes in TorchTune.

.. grid:: 1

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * What are recipes?
      * What are the core components that make up a recipe?
      * How should I structure a new recipe?


What are Recipes?
-----------------
Recipes are the primary entry points for TorchTune users. These can be thought of
as "targeted" end-to-end pipelines for training and optionally evaluating LLMs.
Each recipe implements a training method (eg: full fine-tuning) with a set of meaningful
features (eg: FSDP + Activation Checkpointing + Gradient Accumulation + Mixed Precision training)
applied to a given model family (eg: Llama2).

As model training gets more and more complex, it becomes harder to anticipate new model architectures
and training methodologies while also reasoning about every possible trade-off (eg: memory vs model quality).
We believe a) users are best suited to make trade-offs specific to
their use cases and b) there's no one-size-fits-all solution. As a result, recipes are meant to be easy
to understand, extend and debug, *and not* generalized entry points for all possible settings.

Depending on your use case and level of expertise, you will routinely find yourself modifying existing
recipes (eg: adding new features) or writing new ones. TorchTune makes writing recipes easy by providing
well-tested modular components/building-blocks and general utilities
(eg: :ref:`WandB Logging<metric_logging_label>` and :ref:`FSDP Wrapping <dist_label>`).


Each recipe consists of three components:

- **Configurable parameters**, specified through yaml configs, command-line overrides and dataclasses
- **Recipe Script**, entry-point which puts everything together including parsing and validating configs, setting up the environment, and correctly using the recipe class
- **Recipe Class**, core logic needed for training, exposed to users through a set of APIs

In the following sections, we'll take a closer look at each of these components. For a complete working example, refer to the
`full finetuning recipe <https://github.com/pytorch-labs/torchtune/blob/main/recipes/full_finetune.py>`_ in TorchTune and the associated
`config <https://github.com/pytorch-labs/torchtune/blob/main/recipes/configs/alpaca_llama2_full_finetune.yaml>`_.


What Recipes are not?
---------------------

- **Monolithic Trainers.** A recipe is **not** a monolithic trainer meant to support every possible feature through 100s of flags.
- **Genealized entry-points.** A recipe is **not** meant to support every possible model archichtecture or fine-tuning method.
- **Wrappers around external frameworks.** A recipe is **not** meant to be a wrapper around external frameworks. These are fully written in native-PyTorch using TorchTune building blocks. Dependencies are primarily in the form of additional utilities or interoperability with the surrounding ecosystem (eg: EluetherAI's evaluation harness).


Configs
-------

If you're new to TorchTune or to LLMs generally, configs would be the first concept to understand and get familiar with.
If you're an advanced user writing your own recipes, adding config files will improve your experimentation velocity and
ability to collaborate on experiments.

- TODO - point to config tutorial after this is landed


Recipe Script
-------------

This is the primary entry point for each recipe and provides the user with control over how the recipe is set up, how models are
trained and how the subsequent checkpoints are used. This includes:

- Setting up of the environment
- Parsing and validating configs
- Training the model
- Post-training operations such as evaluation, quantization, model export, generation etc
- Setting up multi-stage training (eg: Distillation) using multiple recipe classes


Scripts should generally structure operations in the following order:

- Extract and validate training params
- Initialize the recipe class which in-turn initializes recipe state
- Load and Validate checkpoint to update recipe state if resuming training
- Initialize recipe components (model, tokenizer, optimizer, loss and dataloader) from checkpoint (if applicable)
- Train the model
- Clean up recipe state after training is complete


An example script looks something like this:

.. code-block:: python

    # Launch using TuneCLI which uses TorchRun under the hood
    parser = utils.TuneArgumentParser(...)

    # Parse and validate the params
    args, _ = parser.parse_known_args()
    args = vars(args)
    recipe_params = FullFinetuneParams(**args)

    # Env variables set by torch run; only need to initialize process group
    init_process_group(backend="nccl")

    # Setup the recipe and train the model
    recipe = FullFinetuneRecipe(params=recipe_params)
    recipe.setup(params=recipe_params)
    recipe.train()
    recipe.cleanup()

    # Other stuff to do after training is complete
    ...


Recipe Class
------------

The recipe class carries the core logic for training a model. Each class implements a relevant interface and exposes a
set of APIs. For fine-tuning, the structure of this class is as follows:

Initialize recipe state including seed, device, dtype, metric loggers, relevant flags etc:

.. code-block:: python

    def __init__(...):

        self._device = utils.get_device(device=params.device)
        self._dtype = utils.get_dtype(dtype=params.dtype)
        ...

Load checkpoint, update recipe state from checkpoint, initialize components and load state dicts from checkpoint

.. code-block:: python

    def setup(...):

        ckpt_dict = self.load_checkpoint(ckpt_path=params.model_checkpoint)

        # If we're resuming from checkpoint, the recipe's state should be updated before
        # initializing the training components.
        if self._resume_from_checkpoint:
            self._update_recipe_state(ckpt_dict)


        # Setup the model, including FSDP wrapping, setting up activation checkpointing and
        # loading the state dict
        self._model = self._setup_model(...)
        self._tokenizer = self._setup_tokenizer(...)

        # Setup Optimizer, including transforming for FSDP when resuming training
        self._optimizer = self._setup_optimizer(...)
        self._loss_fn = self._setup_loss(...)
        self._sampler, self._dataloader = self._setup_data(...)



Run forward and backward across all epochs and save checkpoint at end of each epoch

.. code-block:: python

    def train(...):

        self._optimizer.zero_grad()
        for curr_epoch in range(self.epochs_run, self.total_epochs):

            for idx, batch in enumerate(self._dataloader):
                ...

                with self._autocast:
                    logits = self._model(...)
                    ...
                    loss = self._loss_fn(logits, labels)

                if self.total_training_steps % self._log_every_n_steps == 0:
                    self._metric_logger.log_dict(...)

                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()

                # Update the number of steps when the weights are updated
                self.total_training_steps += 1

            self.save_checkpoint(epoch=curr_epoch)


Cleanup recipe state

.. code-block:: python

    def cleanup(...)

        self.metric_loggers.close()
        ...


Defining parameters for custom recipes
--------------------------------------

In general, you should expose the minimal amount of parameters you need to run and experiment with your recipes.
Exposing an excessive number of parameters will lead to bloated configs, which are more error prone, harder to read, and harder to manage.
On the other hand, hardcoding all parameters will prevent quick experimentation without a code change. Only parametrize what is needed.

Parameters should be organized in a singular dataclass that is passed into the recipe.
This serves as a single source of truth for the details of a fine-tuning run that can be easily validated in code and shared with collaborators for reproducibility.

.. code-block:: python

    class FullFinetuneParams:
        # Model
        model: str = ""
        model_checkpoint: str = ""

In the dataclass, all fields should have defaults assigned to them.
If a reasonable value cannot be assigned or it is a required argument,
use the null value for that data type as the default and ensure that it is set
by the user in the :code:`__post_init__` (see Parameter Validation).
The dataclass should go in the :code:`recipes/params/` folder and the name of
the file should match the name of the recipe file you are creating.

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

Parameter validation
--------------------
To validate user arguments for your dataclass and recipe, use the :code:`__post_init__` method to house any checks and raised exceptions.

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
there should not be any new fields that are not in the dataclass. Any parameters that
are not specified in the config will take on the default value defined in the dataclass.

.. code-block:: yaml

    dataset: alpaca
    seed: null
    shuffle: True
    model: llama2_7b
    ...


Testing configs
---------------
If you plan on contributing your config to the repo, we recommend adding it to the testing suite. TorchTune has testing for every config added to the library, namely ensuring that it instantiates the dataclass and runs the recipe correctly.

To add your config to this test suite, simply update the dictionary in :code:`recipes/tests/configs/test_configs.py`.

.. code-block:: python

    config_to_params = {
        os.path.join(ROOT_DIR, "alpaca_llama2_full_finetune.yaml"): FullFinetuneParams,
        ...,
    }


Command-line overrides
----------------------
To enable quick experimentation, you can specify override values to parameters in your config
via the :code:`tune` command. These should be specified with the flag :code:`--override k1=v1 k2=v2 ...`

The order of overrides from these parameter sources is as follows, with highest precedence first: CLI, Config, Dataclass defaults


Running your recipe
-------------------
If everything is set up correctly, you should be able to run your recipe just like the existing library recipes using the :code:`tune` command:

.. code-block:: bash

    tune <recipe> --config <config> --override ...
