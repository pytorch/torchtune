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
`full finetuning recipe <https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_distributed.py>`_ in TorchTune and the associated
`config <https://github.com/pytorch/torchtune/blob/main/recipes/configs/7B_full.yaml>`_.


What Recipes are not?
---------------------

- **Monolithic Trainers.** A recipe is **not** a monolithic trainer meant to support every possible feature through 100s of flags.
- **Generalized entry-points.** A recipe is **not** meant to support every possible model architecture or fine-tuning method.
- **Wrappers around external frameworks.** A recipe is **not** meant to be a wrapper around external frameworks. These are fully written in native-PyTorch using TorchTune building blocks. Dependencies are primarily in the form of additional utilities or interoperability with the surrounding ecosystem (eg: EleutherAI's evaluation harness).


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
    parser = utils.TuneRecipeArgumentParser(...)

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

Running Recipes with Configs
----------------------------

To run a recipe with a set of user-defined parameters, you will need to write a config file.
You can learn all about configs in our :ref:`config tutorial<config_tutorial_label>`.

Config and CLI parsing using :code:`parse`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We provide a convenient decorator :func:`~torchtune.config._parse.parse` that wraps
your recipe to enable running from the command-line with :code:`tune` with config
and CLI override parsing.

.. code-block:: python

    @config.parse
    def recipe_main(cfg: DictConfig) -> None:
        recipe = FullFinetuneRecipe(cfg=cfg)
        recipe.setup(cfg=cfg)
        recipe.train()
        recipe.cleanup()


Running your recipe
^^^^^^^^^^^^^^^^^^^
You should be able to run your recipe by providing the direct paths to your custom
recipe and custom config using the :code:`tune` command with any CLI overrides:

.. code-block:: bash

    tune <path/to/recipe> --config <path/to/config> k1=v1 k2=v2 ...
