
[![Unit Test](https://github.com/pytorch-labs/torchtune/actions/workflows/unit_test.yaml/badge.svg?branch=main)](https://github.com/pytorch-labs/torchtune/actions/workflows/unit_test.yaml)
![Recipe Integration Test](https://github.com/pytorch-labs/torchtune/actions/workflows/recipe_test.yaml/badge.svg)

# TorchTune (alpha release)

[**Introduction**](#introduction) | [**Installation**](#installation) | [**Get Started**](#get-started) | [**Contributing**](#contributing) |

&nbsp;

## Introduction

TorchTune is a native-Pytorch library for easily authoring, fine-tuning and experimenting with LLMs.

The library provides:

- Native-PyTorch implementations of popular LLMs, with convertors to transform checkpoints into TorchTune's format
- Training recipes for popular fine-tuning techniques with reference benchmarks and comprehensive correctness checks
- Integration with HuggingFace Datasets for training and EleutherAI's Eval Harness for evaluation
- Support for distributed training using FSDP from PyTorch Distributed
- Yaml configs for easily configuring training runs

NOTE: TorchTune is currently only tested with the latest stable PyTorch release, which is currently [2.2.0](https://pytorch.org/get-started/locally/).

&nbsp;

| Model                                         | Sizes     |   Finetuning Methods |
|-----------------------------------------------|-----------|-----------------------------------------------------------|
| [Llama2](torchtune/models/llama2.py)   | 7B        | [Full Finetuning](recipes/full_finetune.py), [LoRA]()  |

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

---

## Installation

Currently, `torchtune` must be built via cloning the repository and installing as follows:

```
git clone https://github.com/pytorch-labs/torchtune.git
cd torchtune
pip install -e .
```

To confirm that the package is installed correctly, you can run the following command:

```
tune recipe --help
```

And should see the following output:

```
usage: tune recipe

Utility for information relating to recipes

positional arguments:

    list      List recipes
    cp        Copy recipe to local path

options:
  -h, --help  show this help message and exit
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
tune convert_checkpoint --checkpoint-path <CHECKPOINT_PATH>
```

&nbsp;

#### Running recipes

On a single GPU
```
tune finetune_llm --config alpaca_llama2_finetune
```

On multiple GPUs using FSDP
```
tune --nnodes 1 --nproc_per_node 4 finetune_llm --config alpaca_llama2_finetune --fsdp True
```

&nbsp;

#### Copy and edit a custom recipe

To copy a recipe to customize it yourself and then run
```
tune recipe cp finetune_llm my_recipe/finetune_llm.py
tune config cp alpaca_llama2_finetune my_recipe/alpaca_llama2_finetune.yaml
tune my_recipe/finetune_llm.py --config my_recipe/alpaca_llama2_finetune.yaml
```

&nbsp;

#### Command Utilities

``tune`` provides functionality for launching torchtune recipes as well as local
recipes. Aside from torchtune recipe utilties, it integrates with ``torch.distributed.run``
to support distributed job launching by default. ``tune`` offers everyting that ``torchrun``
does with the following additional functionalities:

1. ``tune <recipe> <recipe_args>`` with no optional ``torchrun`` options launches a single python process

2. ``<recipe>`` and recipe arg ``<config>`` can both be passed in as names instead of paths if they're included in torchtune

3. ``tune <path/to/recipe.py> <recipe_args>`` can be used to launch local recipes

4. ``tune <torchrun_options> <recipe> <recipe_args>`` will launch a torchrun job

5. ``tune recipe`` and ``tune config`` commands provide utilities for listing and copying packaged recipes and configs

&nbsp;

---

## Contributing

We welcome any feature requests, bug reports, or pull requests from the community. See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.
