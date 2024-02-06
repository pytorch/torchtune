# Training Recipe Deep-dive

This tutorial will walk you through the design of training-recipes in TorchTune and answer the following questions:

- What are recipes and how are these structured?
- What can I do with existing recipes?
- How should I modify a given recipe to write my own?

For information on how to launch a training run, please look at ["Getting Started"](../README.md#get-started) and the [Recipes README](../recipes/README.md).

&nbsp;

## What are recipes?

Recipes are the primary entry points for TorchTune users. These can be thought of as "targeted" end-to-end pipelines for training and optionally evaluating LLMs. Each recipe implements a training method (eg: full fine-tuning) with a set of meaningful features (eg: FSDP + Activation Checkpointing + Gradient Accumulation + Mixed Precision training) applied to a given model family (eg: Llama2).

As model training gets more and more complex, it becomes harder to anticipate new model architectures and training methodologies while also reasoning about every possible trade-off (eg: memory vs model quality). We believe users are best suited to make trade-offs specific to their use cases and that there's no one-size-fits-all solution. As a result, recipes are meant to be easy to understand, extend and debug *instead of* generalized entry points for many settings.

Depending on their use case and level of expertise, users will routinely find themselves modifying existing recipes (eg: adding new features) or writing new ones. TorchTune makes writing recipes easy by providing well-tested modular components/building-blocks and general utilities (eg: [WandB logging](../torchtune/utils/metric_logging.py)).

Each recipe consists of three components:

- **Configurable parameters**, specified through yaml configs [example](../recipes/configs/alpaca_llama2_full_finetune.yaml), command-line overrides and dataclasses [example](../recipes/params.py)
- **Recipe Class**, core logic needed for training, exposed to users through a set of APIs [interface](../recipes/interfaces.py)
- **Recipe Script**, entry-point which puts everything together including parsing and validating configs, setting up the environment, and correctly using the recipe class

In the following sections, we'll take a closer look at each of these components.

&nbsp;

### What recipes are not?

- Monolithic Trainers. A recipe is **not** a monolithic trainer meant to support every possible feature through 100s of flags.
- Genealized entry-points. A recipe is **not** meant to support every possible model archichtecture or fine-tuning method.
- Wrappers around external frameworks. A recipe is **not** meant to be a wrapper around external frameworks. These are fully written in native-PyTorch and dependencies are primarily in the form of utilities.

&nbsp;

## Configs

If you're new to TorchTune or to LLMs generally, configs would be the first concept to understand and get familiar with. If you're an advanced user writing your own recipes, adding config files will improve your experimentation velocity and ability to collaborate on experiments.

For more information on the structure of TorchTune configs, refer to the [Recipes README](../recipes/README.md)
- TODO - point to config tutorial after this is landed

&nbsp;

## Recipe Script

This is the primary entry point for each recipe and provides the user with control over how the recipe is setup, model(s) is(are) trained and how the subsequent checkpoints are used. This includes:
- Setting up of the environment
- Parsing and validating configs
- Initializing and setting up the recipe class
- Training the model
- Post-training operations such as evaluation, quantization, model export, generation etc
- Setting up multi-stage training (eg: Distillation) using multiple Recipe classes

&nbsp;

Scripts should generally structure operations in the following order:

- Extract and validate training params
- Intialize [Recipe Class](#recipe-class) which in-turn intializes recipe state
- Load and Validate checkpoint to update recipe state if resuming training
- Initialize recipe components (model, tokeinzer, optimizer, loss and dataloader) from checkpoint (if applicable)
- Train the model, with checkpoints at the end of every epoch
- Clean up recipe state after training is complete

&nbsp;

Example script for [full fine-tuning](../recipes/full_finetune.py):

```
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
```

&nbsp;

## Recipe Class

The Recipe Class carries the core logic for training a model. Each class implements a relevant [interface](../recipes/interfaces.py) and exposes a set of APIs used to setup training in the Recipe Main. For fine-tuning, the structure of this class [[full finetune example](../recipes/full_finetune.py)] is as follows:

&nbsp;

```__init__(...)```
-   Initialize recipe state including seed, device, dtype, metric loggers, relevant flags etc

```
self._device = utils.get_device(device=params.device)
self._dtype = utils.get_dtype(dtype=params.dtype)
...
```

&nbsp;

```setup(...)```:
-   Load checkpoint from specified path; validatie the checkpoint and updating recipe state if resuming training
-   Setup the model, including FSDP wrapping, setting up activation checkpointing and loading the state dict
-   Load tokenizer
-   Setup Loss
-   Initialize Dataloader and Sampler

```
# Extract state dict from checkpoint
ckpt_dict = self.load_checkpoint(ckpt_path=params.model_checkpoint)


# If we're resuming from checkpoint, the recipe's state should be updated before
# initializing the training components.
if self._resume_from_checkpoint:
    self._update_recipe_state(ckpt_dict)


# Initialize model, load state_dict, enable FSDP and activation checkpointing
self._model = self._setup_model(...)


# Load tokenizer
self._tokenizer = self._setup_tokenizer(...)


# Setup Optimizer, including transforming for FSDP when resuming training
self._optimizer = self._setup_optimizer(...)


self._loss_fn = self._setup_loss(...)
self._sampler, self._dataloader = self._setup_data(...)

```

&nbsp;

```train(...)```:
-   Forward and backward across all epochs
-   Save checkpoint at end of each epoch; checkpoints created during training include optimizer and recipe state

```
# General training loop structure

self._optimizer.zero_grad()
for curr_epoch in range(self.epochs_run, self.total_epochs):

    for idx, batch in enumerate(
        pbar := tqdm(self._dataloader, disable=not (rank == 0))
    ):
        ...

        with self._autocast:
            logits = self._model(input_ids)
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
```

&nbsp;

```cleanup(...)```:
-   Cleanup recipe state

&nbsp;

#### How do I write my own recipe?

Before writing a new recipe, check the [recipes folder](../recipes/) to see if an existing recipe satisfies your use case. If not, following are some common scenarions in which you might need to write some code.

&nbsp;

**Adding a new dataset**
- Add a new dataset to [datasets](../torchtune/datasets/)
- Add the new dataset and associated params to the [params dataclass](../recipes/params.py)
- If needed:
    - Clone the recipe into a new file
    - Update the ```_setup_data``` method to configure the dataloader
    - Update the ```train``` method to read the samples/batches correctly

&nbsp;

**Adding a new model**
- Add a new model to [models](../torchtune/models/) with associated building blocks in [modules](../torchtune/modules/). More details in [this tutorial](../tutorials/)
- If needed:
    - Clone the recipe into a new file
    - Update the ```_setup_model``` method to correctly instantiate model and load the state dict
    - Update the ```train``` method to call ```forward``` correctly

&nbsp;

**Adding a new training method**
- TODO: Update this section after LoRA Recipe lands
