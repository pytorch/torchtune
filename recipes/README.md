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

You can set parameters for a recipe by modifying the yaml config file directly or by providing command-line overrides. Use the `tune` CLI to provide your updated config filepath with the `--config` flag and any overrides in the format `key=value`.

```
tune full_finetune --config path/to/my_config.yaml k1=v1 k2=v2 ...
```

The config is the primary entry point for users, with CLI overrides providing flexibility for quick experimentation.

## Examples

If you have not already done so, follow the instructions [here](https://github.com/pytorch-labs/torchtune/blob/main/README.md#downloading-a-model) to download the Llama2 model and convert the weights.

### Full finetune

To run the `full_finetune` recipe with the `alpaca_llama2_full_finetune.yaml` config on four GPUs with FSDP, run this command:

```
tune --nnodes 1 --nproc_per_node 4 finetune_llm --config alpaca_llama2_finetune
```

### LoRA finetune

We provide separate recipes for finetuning LoRA on a single device and multiple GPUs. You can finetune LoRA on a single device using the `lora_finetune_single_device` recipe with the `alpaca_llama2_lora_finetune_single_device.yaml` config. To do so on multiple GPUs, use `lora_finetune_distributed ` with `alpaca_llama2_lora_finetune_distributed.yaml`. E.g. on two devices, you can run the following:

```
tune --nnodes 1 --nproc_per_node 2 lora_finetune_distributed --config alpaca_llama2_lora_finetune_distributed
```

For both recipes, activation checkpointing is enabled by default, and LoRA weights are added to the Q and V projections in self-attention. FSDP is enabled by default for
distributed recipe. If you additionally want to apply LoRA to K and would like to reduce the LoRA rank from the default of eight, you can run

```
tune --nnodes 1 --nproc_per_node 2 lora_finetune_distributed --config alpaca_llama2_lora_finetune_distributed lora_attn_modules=q_proj,k_proj,v_proj lora_rank=4
```

### Generation

To run the generation recipe, run this command from inside the main `/torchtune` directory:
```
python -m recipes.alpaca_generate --native-checkpoint-path /tmp/finetune-llm/model_0.ckpt --tokenizer-path ~/llama/tokenizer.model --input "What is some cool music from the 1920s?"
```
