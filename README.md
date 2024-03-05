
[![Unit Test](https://github.com/pytorch-labs/torchtune/actions/workflows/unit_test.yaml/badge.svg?branch=main)](https://github.com/pytorch-labs/torchtune/actions/workflows/unit_test.yaml)
![Recipe Integration Test](https://github.com/pytorch-labs/torchtune/actions/workflows/recipe_test.yaml/badge.svg)

# TorchTune (alpha release)

[**Introduction**](#introduction) | [**Installation**](#installation) | [**Get Started**](#get-started) | [**Design Principles**](#design-principles) | [**Contributing**](#contributing) | [**License**](#license)

&nbsp;

## Introduction

TorchTune is a native-Pytorch library for easily authoring, fine-tuning and experimenting with LLMs.

The library provides:

- Native-PyTorch implementations of popular LLMs, with convertors to transform checkpoints into TorchTune's format
- Training recipes for popular fine-tuning techniques with reference benchmarks and comprehensive correctness checks
- Integration with HuggingFace Datasets for training and EleutherAI's Eval Harness for evaluation
- Support for distributed training using FSDP from PyTorch Distributed
- Yaml configs for easily configuring training runs

&nbsp;

The library currently supports the following models and fine-tuning methods.

| Model                                         | Sizes     |   Finetuning Methods |
|-----------------------------------------------|-----------|-----------------------------------------------------------|
| [Llama2](torchtune/models/llama2.py)   | 7B        | [Full Finetuning](recipes/full_finetune.py), [LoRA](recipes/lora_finetune.py)  |

&nbsp;

### Finetuning resource requirements

Note: These resource requirements are based on GPU peak memory reserved during training using the specified configs. You may
experience different peak memory utilization based on changes made in configuration / training. Please see the linked configs in the table for specific settings such as batch size, FSDP, activation checkpointing, optimizer, etc used to obtain the peak memory.

| HW Resources | Finetuning Method |  Config | Model Size | Peak Memory per GPU
|--------------|-------------------|---------|------------|---------------------|
| 2 x RTX 4090 |     LoRA          | [lora_finetune](https://github.com/pytorch-labs/torchtune/blob/main/recipes/configs/alpaca_llama2_lora_finetune.yaml)    |    7B      |    18 GB *           |
| 1 x A6000    |     LoRA          | [lora_finetune](https://github.com/pytorch-labs/torchtune/blob/main/recipes/configs/alpaca_llama2_lora_finetune.yaml)    |    7B      |    29.5 GB *           |
| 4 x T4       |     LoRA          | [lora_finetune](https://github.com/pytorch-labs/torchtune/blob/main/recipes/configs/alpaca_llama2_lora_finetune.yaml)    |    7B      |    12 GB *           |
| 2 x A100 80G |   Full finetune   | [full_finetune](https://github.com/pytorch-labs/torchtune/blob/main/recipes/configs/alpaca_llama2_full_finetune.yaml)    |    7B      |    62 GB             |
| 8 x A6000    |   Full finetune   | [full_finetune](https://github.com/pytorch-labs/torchtune/blob/main/recipes/configs/alpaca_llama2_full_finetune.yaml)    |    7B      |    42 GB *             |


NOTE: * indicates an estimated metric based on experiments conducted on A100 GPUs with GPU memory artificially limited using [torch.cuda.set_per_process_memory_fraction API](https://pytorch.org/docs/stable/generated/torch.cuda.set_per_process_memory_fraction.html). Peak memory per GPU is as reported by `nvidia-smi` monitored over a couple hundred training iterations. Please file an issue if you are not able to reproduce these results when running TorchTune on certain hardware.

&nbsp;

---

## Installation

Currently, `torchtune` must be built via cloning the repository and installing as follows:

NOTE: TorchTune is currently only tested with the latest stable PyTorch release, which is currently [2.2](https://pytorch.org/get-started/locally/).

```
git clone https://github.com/pytorch-labs/torchtune.git
cd torchtune
pip install -e .
```

To confirm that the package is installed correctly, you can run the following command:

```
tune
```

And should see the following output:

```
usage: tune [options] <recipe> [recipe_args]
tune: error: the following arguments are required: recipe, recipe_args
```

&nbsp;

---

## Get Started

For our quickstart guide to getting you finetuning an LLM fast, see our [Finetuning Llama2 with TorchTune](https://torchtune-preview.netlify.app/examples/first_finetune_tutorial.html) tutorial. You can also follow the steps below.

#### Downloading a model

Follow the instructions on the official [`meta-llama`](https://huggingface.co/meta-llama/Llama-2-7b) repository to ensure you have access to the Llama2 model weights. Once you have confirmed access, you can run the following command to download the weights to your local machine. This will also download the tokenizer model and a responsible use guide.

> Set your environment variable `HF_TOKEN` or pass in `--hf-token` to the command in order to validate your access.
You can find your token at https://huggingface.co/settings/tokens

```
tune download --repo-id meta-llama/Llama-2-7b --hf-token <HF_TOKEN> --output-dir /tmp/llama2
```

&nbsp;

#### Converting the checkpoint into PyTorch-native

Now that you have the Llama2 model weights, convert them into a PyTorch-native format supported by TorchTune.

```
tune convert_checkpoint --checkpoint-path /tmp/llama2/consolidated.00.pth --output-path /tmp/llama2_native --model llama2
```

&nbsp;

#### Running recipes

TorchTune contains recipes for [full finetuning](https://github.com/pytorch-labs/torchtune/blob/e802c057d17773f65cf80721807086724e4fa7db/recipes/full_finetune.py), [LoRA finetuning](https://github.com/pytorch-labs/torchtune/blob/e802c057d17773f65cf80721807086724e4fa7db/recipes/lora_finetune.py), and [generation](https://github.com/pytorch-labs/torchtune/blob/e802c057d17773f65cf80721807086724e4fa7db/recipes/alpaca_generate.py).

To run a full finetune on two devices on the Alpaca dataset using FSDP:

```
tune --nnodes 1 --nproc_per_node 2 full_finetune --config alpaca_llama2_full_finetune
```

The argument passed to `--nproc_per_node` can be varied depending on how many GPUs you have. A full finetune can be memory-intensive, so make sure you are running on enough devices. See [this table](https://github.com/pytorch-labs/torchtune/blob/main/README.md#finetuning-resource-requirements) for resource requirements on common hardware setups.

Similarly, you can finetune with LoRA on the Alpaca dataset on two devices via

```
tune --nnodes 1 --nproc_per_node 2 lora_finetune --config alpaca_llama2_lora_finetune
```

Again, the argument to `--nproc_per_node` can be varied subject to memory constraints of your device(s).

&nbsp;

#### Copy and edit a custom recipe

To copy a recipe to customize it yourself and then run
```
tune cp full_finetune.py my_recipe/full_finetune.py
tune cp alpaca_llama2_full_finetune.yaml my_recipe/alpaca_llama2_full_finetune.yaml
tune my_recipe/full_finetune.py --config my_recipe/alpaca_llama2_full_finetune.yaml
```

&nbsp;

#### Command Utilities

``tune`` provides functionality for launching torchtune recipes as well as local
recipes. Aside from torchtune recipe utilties, it integrates with ``torch.distributed.run``
to support distributed job launching by default. ``tune`` offers everyting that ``torchrun``
does with the following additional functionalities:

1. ``tune <torchrun_options> <recipe> <recipe_args>`` will launch a torchrun job

2. ``<recipe>`` and recipe arg ``<config>`` can both be passed in as names instead of paths if they're included in torchtune

3. ``tune ls`` and ``tune cp`` commands provide utilities for listing and copying packaged recipes and configs

&nbsp;

---

## Design Principles

TorchTune embodies PyTorch’s design philosophy [[details](https://pytorch.org/docs/stable/community/design.html)], especially "usability over everything else".

#### Native PyTorch

TorchTune is a native-PyTorch library. While we provide integrations with the surrounding ecosystem (eg: HuggingFace Datasets, EluetherAI Eval Harness), all of the core functionality is written in PyTorch.

#### Simplicity and Extensibility

TorchTune is designed to be easy to understand, use and extend.

- Composition over implementation inheritance - layers of inheritance for code re-use makes the code hard to read and extend
- No training frameworks - explicitly outlining the training logic makes it easy to extend for custom use cases
- Code duplication is prefered over unecessary abstractions
- Modular building blocks over monolithic components

#### Correctness

TorchTune provides well-tested components with a high-bar on correctness. The library will never be the first to provide a feature, but available features will be thoroughly tested. We provide

- Extensive unit-tests to ensure component-level numerical parity with reference implementations
- Checkpoint-tests to ensure model-level numerical parity with reference implementations
- Integration tests to ensure recipe-level performance parity with reference implementations on standard benchmarks

&nbsp;

## Contributing

We welcome any feature requests, bug reports, or pull requests from the community. See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

&nbsp;

## License

TorchTune is released under the [BSD 3 license](./LICENSE).
