
# torchtune

**Note: This repository is currently under heavy development.**

The torchtune package contains tools and utilities to finetune generative models with native PyTorch techniques.

# Unit Tests Status

[![Unit Test](https://github.com/pytorch-labs/torchtune/actions/workflows/unit_test.yaml/badge.svg?branch=main)](https://github.com/pytorch-labs/torchtune/actions/workflows/unit_test.yaml)

# Recipe Integration Test Status

![Recipe Integration Test](https://github.com/pytorch-labs/torchtune/actions/workflows/recipe_integration_test.yaml/badge.svg)

# Installation

This library requires PyTorch >= 2.0. Please install locally using [this guide](https://pytorch.org/get-started/locally/).

Currently, `torchtune` must be built via cloning the repository and installing as follows:

```
git clone https://github.com/pytorch-labs/torchtune
cd torchtune
pip install -e .
```

To verify successful installation, one can run:

```
tune recipe list
```

And as an example, the following import should work:

```
from torchtune.modules import TransformerDecoder
```

# Quickstart

### Downloading a model

Follow the instructions on the official [`meta-llama`](https://huggingface.co/meta-llama/Llama-2-7b) repository
to ensure you have access to the Llama2 model weights. Once you have confirmed access, you can run the following command to download the weights to your local machine.

> Set your environment variable `HF_TOKEN` or pass in `--hf-token` to the command in order to validate your access. You can find your token at https://huggingface.co/settings/tokens

```
tune download --repo-id meta-llama/Llama-2-7b --hf-token <HF_TOKEN>
```

### Converting the checkpoint into PyTorch-native format

Now that you have the Llama2 model weights, we need to convert into a PyTorch-native format supported by Torchtune.

```
tune convert_checkpoint --checkpoint-path <CHECKPOINT_PATH>
```

### Running recipes

On a single GPU
```
tune finetune_llm --config alpaca_llama2_finetune
```

On multiple GPUs using FSDP
```
tune --nnodes 1 --nproc_per_node 4 finetune_llm --config alpaca_llama2_finetune --fsdp True
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

# Contributing

We welcome your contribution! Please see our [Contribution Guide](./CONTRIBUTING.md) for more details on how to effectively
develop with Torchtune.
