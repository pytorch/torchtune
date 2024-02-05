# Training Recipes

&nbsp;

## What are Recipes?

Recipes are the primary entry points for TorchTune users. These can be thought of as end-to-end pipelines for training and optionally evaluating LLMs. Each recipe consists of three components:

- **Configurable parameters**, specified through yaml configs [example](https://github.com/pytorch-labs/torchtune/blob/main/recipes/configs/alpaca_llama2_full_finetune.yaml), command-line overrides and dataclasses
- **Recipe class**, core logic needed for training, exposed to users through a set of APIs [interface](https://github.com/pytorch-labs/torchtune/blob/main/recipes/interfaces.py)
- **Recipe script**, puts everything together including parsing and validating configs, setting up the environment, and correctly using the recipe class

&nbsp;

## Recipe Design

Recipes in TorchTune are:

1. **Simple**. Written fully in native-PyTorch.
2. **Correct**. Numerical parity verification for every component and extensive comparisons with reference implementations and benchmarks.
3. **Easy to Understand**. Each recipe provides a limited set of meaningful features, instead of every possible feature hidden behind 100s of flags. Code duplication is preferred over unnecessary abstractions.
4. **Easy to Extend**. No dependency on training frameworks and no implementation inheritance. Users don't need to go through layers-upon-layers of abstractions to figure out how to extend core functionality.
5. **Accessible to a spectrum of Users**. Users can decide how they want to interact with TorchTune Recipes:
    - Start training models by modifying existing configs
    - Modify existing recipes for custom cases
    - Directly use available building blocks to write completely new recipes/training paradigms

&nbsp;

## How to specify parameters for recipes

The arguments for a recipe are defined in a params object (such as `FullFinetuneParams`) that contains the full list of configurable parameters. These are either set to default values or sourced from the YAML file listed with `--config` and `--override` arguments in the `tune` CLI. The `TuneArgumentParser` class is responsible for parsing the provided config file and overrides and funneling it into the corresponding params object for the recipe the user wishes to run. The order of overrides from these parameter sources is as follows, with highest precedence first:

CLI &rarr; Config &rarr; Params defaults

The config is the primary entry point for users, with CLI overrides providing flexibility for quick experimentation.

## Examples

To run the `finetune_llm` recipe with the `alpaca_llama2_finetune.yaml` config, run this command:

On GPU (without PyTorch Distributed):
```
tune finetune_llm --config alpaca_llama2_finetune --override device=cuda
```

On multiple GPUs with FSDP:
```
tune --nnodes 1 --nproc_per_node 4 finetune_llm --config alpaca_llama2_finetune --override enable_fsdp=True enable_activation_checkpointing=False device=cuda
```

To run the generation recipe, run this command from inside the main `/torchtune` directory:
```
python -m recipes.alpaca_generate --native-checkpoint-path /tmp/finetune-llm/model_0.ckpt --tokenizer-path ~/llama/tokenizer.model --input "What is some cool music from the 1920s?"
```
