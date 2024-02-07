.. _recipe_deepdive:

=========================
Training Recipe Deep-Dive
=========================

This tutorial will walk you through the design of training-recipes in TorchTune.

.. grid:: 1

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * What are recipes?
      * How should I structure my own recipe?


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

In the following sections, we'll take a closer look at each of these components.


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

This is the primary entry point for each recipe and provides the user with control over how the recipe is setup, how models are
trained and how the subsequent checkpoints are used. This includes:

- Setting up of the environment
- Parsing and validating configs
- Training the model
- Post-training operations such as evaluation, quantization, model export, generation etc
- Setting up multi-stage training (eg: Distillation) using multiple Recipe classes


Scripts should generally structure operations in the following order:

- Extract and validate training params
- Intialize th Recipe Class which in-turn intializes recipe state
- Load and Validate checkpoint to update recipe state if resuming training
- Initialize recipe components (model, tokeinzer, optimizer, loss and dataloader) from checkpoint (if applicable)
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

The Recipe Class carries the core logic for training a model. Each class implements a relevant and exposes a
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



Run Forward and backward across all epochs and save checkpoint at end of each epoch

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
