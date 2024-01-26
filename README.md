
# Torchtune

[![Unit Test](https://github.com/pytorch-labs/torchtune/actions/workflows/unit_test.yaml/badge.svg?branch=main)](https://github.com/pytorch-labs/torchtune/actions/workflows/unit_test.yaml)

![Recipe Integration Test](https://github.com/pytorch-labs/torchtune/actions/workflows/recipe_integration_test.yaml/badge.svg)

The Torchtune package contains tools and utilities to finetune generative models with native PyTorch techniques.

**🚧 This repository is currently under heavy development 🚧**

## Why Torchtune?

## Quickstart

### Install

This library requires PyTorch >= 2.0. Please install locally using [this guide](https://pytorch.org/get-started/locally/).

Then, install using `Pip`.
```
pip install torchtune
```

To verify successful installation, one can run:

```
tune recipe list
```

### Running recipes

On a single GPU
```
tune finetune_llm --config alpaca_llama2_finetune
```

On multiple GPUs using FSDP
```
tune --nnodes 1 --nproc_per_node 4 finetune_llm --config alpaca_llama2_finetune
```

### Copy and edit a custom recipe

To copy a recipe to customize it yourself and then run
```
tune recipe cp finetune_llm my_recipe/finetune_llm.py
tune config cp alpaca_llama2_finetune my_recipe/alpaca_llama2_finetune.yaml
tune my_recipe/finetune_llm.py --config my_recipe/alpaca_llama2_finetune.yaml
```

### Command Utilities

``tune`` provides functionality for launching torchtune recipes as well as local
recipes. Aside from torchtune recipe utilties, it integrates with ``torch.distributed.run``
to support distributed job launching by default. ``tune`` offers everyting that ``torchrun``
does with the following additional functionalities:

1. ``tune <recipe> <recipe_args>`` with no optional ``torchrun`` options launches a single python process

2. ``<recipe>`` and recipe arg ``<config>`` can both be passed in as names instead of paths if they're included in torchtune

3. ``tune <path/to/recipe.py> <recipe_args>`` can be used to launch local recipes

4. ``tune <torchrun_options> <recipe> <recipe_args>`` will launch a torchrun job

5. ``tune recipe`` and ``tune config`` commands provide utilities for listing and copying packaged recipes and configs


## Supported models and datasets

|Model|Size|Reference|
|----|----|----|
|Llama2|7B|https://ai.meta.com/llama/|

|Dataset|Samples|Reference|
|----|----|----|
|Alpaca|52k|https://github.com/tatsu-lab/stanford_alpaca|
|SlimOrca|518k|https://huggingface.co/datasets/Open-Orca/SlimOrca|

## Debugging & Getting Help

Here's a bunch of resources to help you if you get stuck!
- [Why is my model OOM?](https://google.com)
- [Common errors in Torchtune](https://google.com)

And see all of our tutorials at [...]

## Contributing

We welcome any and all contributions! See our [Contributing Guide](./CONTRIBUTING.md) for how to get started.

## Citation

## License
