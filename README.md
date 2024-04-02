
[![Unit Test](https://github.com/pytorch/torchtune/actions/workflows/unit_test.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtune/actions/workflows/unit_test.yaml)
![Recipe Integration Test](https://github.com/pytorch/torchtune/actions/workflows/recipe_test.yaml/badge.svg)
[![](https://dcbadge.vercel.app/api/server/4Xsdn8Rr9Q?style=flat)](https://discord.gg/4Xsdn8Rr9Q)

# TorchTune is still pre-release!

TorchTune is currently in a pre-release state and under extensive development.
While the 0.0.1 version is available on PyPI there may still be rough edges.
Stay *tuned* for the first release in the coming weeks!


# TorchTune

[**Introduction**](#introduction) | [**Installation**](#installation) | [**Get Started**](#get-started) |  [**Documentation**](https://pytorch.org/torchtune) | [**Design Principles**](#design-principles) | [**Contributing**](#contributing) | [**License**](#license)

&nbsp;

## Introduction

TorchTune is a native-Pytorch library for easily authoring, fine-tuning and experimenting with LLMs.

The library provides:

- Native-PyTorch implementations of popular LLMs
- Support for checkpoints in various formats, including checkpoints in HF format
- Training recipes for popular fine-tuning techniques with reference benchmarks and comprehensive correctness checks
- Evaluation of trained models with EleutherAI Eval Harness
- Integration with Hugging Face Datasets for training
- Support for distributed training using FSDP from PyTorch Distributed
- YAML configs for easily configuring training runs
- [Upcoming] Support for lower precision dtypes and quantization techniques from [TorchAO](https://github.com/pytorch-labs/ao)
- [Upcoming] Interop with various inference engines

&nbsp;

The library currently supports the following models and fine-tuning methods.

| Model                                         | Sizes     |   Finetuning Methods |
|-----------------------------------------------|-----------|-----------------------------------------------------------|
| [Llama2](torchtune/models/llama2/_model_builders.py)   | 7B        | Full Finetuning [[single device](recipes/configs/llama2/7B_full_single_device.yaml),  [distributed](recipes/configs/llama2/7B_full.yaml)] LoRA [[single device](recipes/configs/llama2/7B_lora_single_device.yaml),  [distributed](recipes/configs/llama2/7B_lora.yaml)] QLoRA [single device](recipes/configs/llama2/7B_qlora_single_device.yaml) |
| [Llama2](torchtune/models/llama2/_model_builders.py)   | 13B       | [Full Finetuning](recipes/configs/llama2/13B_full.yaml), [LoRA](recipes/configs/llama2/13B_lora.yaml)
| [Mistral](torchtune/models/mistral//_model_builders.py)   | 7B       | [Full Finetuning](recipes/configs/mistral/7B_full.yaml), [LoRA](recipes/configs/mistral/7B_lora.yaml)


&nbsp;

### Finetuning resource requirements

Note: These resource requirements are based on GPU peak memory reserved during training using the specified configs. You may
experience different peak memory utilization based on changes made in configuration / training. Please see the linked configs in the table for specific settings such as batch size, FSDP, activation checkpointing, optimizer, etc used to obtain the peak memory. The specific HW resources specified are meant as an example for possible hardware that can be used.

| Example HW Resources | Finetuning Method |  Config | Model | Peak Memory per GPU
|--------------|-------------------|---------|------------|---------------------|
| 1 x RTX 4090 |     QLoRA          | [qlora_finetune_single_device](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_qlora_single_device.yaml)         |    Llama-7B      |     9.29 GB *           |
| 2 x RTX 4090 |     LoRA          | [lora_finetune_distributed](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_lora.yaml)         |    Llama-7B      |    14.17 GB *           |
| 1 x RTX 4090 |     LoRA          | [lora_finetune_single_device](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_lora_single_device.yaml)     |    Llama-7B      | 17.18 GB *           |
| 1 x RTX 4090 |   Full finetune   | [full_finetune_single_device](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full_single_device_low_memory.yaml)     |    Llama-7B      |    15.97 GB * ^           |
| 4 x RTX 4090 |   Full finetune   | [full_finetune_distributed](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full.yaml)         |    Llama-7B      |    12.01 GB *           |


NOTE: * indicates an estimated metric based on experiments conducted on A100 GPUs with GPU memory artificially limited using [torch.cuda.set_per_process_memory_fraction API](https://pytorch.org/docs/stable/generated/torch.cuda.set_per_process_memory_fraction.html). Peak memory per GPU is as reported by `torch.cuda.max_memory_reserved()`. Please file an issue if you are not able to reproduce these results when running TorchTune on certain hardware.

NOTE: ^ indicates the required use of third-party dependencies that are not installed with ``torchtune`` by default. In particular, for the most memory efficient full finetuning [configuration](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full_single_device_low_memory.yaml), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) is required and can be installed via `pip install bitsandbytes`, after which the configuration
can be run successfully.

&nbsp;

---

## Installation

Currently, `torchtune` must be built via cloning the repository and installing as follows:

NOTE: TorchTune is currently only tested with the latest stable PyTorch release, which is currently [2.2](https://pytorch.org/get-started/locally/).

```
git clone https://github.com/pytorch/torchtune.git
cd torchtune
pip install -e .
```

To confirm that the package is installed correctly, you can run the following command:

```
tune --help
```

And should see the following output:

```
usage: tune [-h] {ls,cp,download,run,validate} ...

Welcome to the TorchTune CLI!

options:
  -h, --help            show this help message and exit

...
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
tune download meta-llama/Llama-2-7b \
--output-dir /tmp/llama2 \
--hf-token <HF_TOKEN> \
```

Note: While the ``tune download`` command allows you to download *any* model from the hub, there's no guarantee that the model can be finetuned with TorchTune. Currently supported models can be found [here](#introduction)

&nbsp;

#### Running recipes

TorchTune contains built-in recipes for:
- Full finetuning on [single device](https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_single_device.py) and on [multiple devices with FSDP](https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_distributed.py)
- LoRA finetuning on [single device](https://github.com/pytorch/torchtune/blob/main/recipes/lora_finetune_single_device.py) and on [multiple devices with FSDP](https://github.com/pytorch/torchtune/blob/main/recipes/lora_finetune_distributed.py).
- QLoRA finetuning on [single device](https://github.com/pytorch/torchtune/blob/main/recipes/lora_finetune_single_device.py), with a QLoRA specific [configuration](https://github.com/pytorch/torchtune/blob/main/recipes/configs/7B_qlora_single_device.yaml)

To run a LoRA finetune on a single device using the [Alpaca Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca):
```
tune run lora_finetune_single_device --config llama2/7B_lora_single_device
```

TorchTune integrates with [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) for easily running distributed training. See below for an example of running a Llama2 7B full-finetune on two GPUs.

> Make sure to place any torchrun commands **before** the recipe specification b/c any other CLI args will
overwrite the config, not affect distributed training.

```
tune run --nproc_per_node 2 full_finetune_distributed --config llama2/7B_full_distributed
```

You can easily overwrite some config properties as follows, but you can also easily copy a built-in config and
modify it following the instructions in the [next section](#copy-and-edit-a-custom-recipe-or-config).
```
tune run lora_finetune_single_device --config llama2/7B_lora_single_device batch_size=8
```

&nbsp;

#### Copy and edit a custom recipe or config

```
tune cp full_finetune_distributed my_custom_finetune_recipe.py
Copied to ./my_custom_finetune_recipe.py

tune cp llama2/7B_full .
Copied to ./7B_full.yaml
```

Then, you can run your custom recipe by directing the `tune run` command to your local files:
```
tune run my_custom_finetune_recipe.py --config 7B_full.yaml
```

&nbsp;

Check out `tune --help` for all possible CLI commands and options.

&nbsp;

---

## Design Principles

TorchTune embodies PyTorch’s design philosophy [[details](https://pytorch.org/docs/stable/community/design.html)], especially "usability over everything else".

#### Native PyTorch

TorchTune is a native-PyTorch library. While we provide integrations with the surrounding ecosystem (eg: Hugging Face Datasets, EluetherAI Eval Harness), all of the core functionality is written in PyTorch.

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

## Acknowledgements

The Llama2 code in this repository is inspired by the original [Llama2 code](https://github.com/meta-llama/llama/blob/main/llama/model.py). We'd also like to give a huge shoutout to some awesome libraries and tools in the ecosystem!

- EleutherAI's [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Hugging Face for the [Datasets Repository](https://github.com/huggingface/datasets)
- [gpt-fast](https://github.com/pytorch-labs/gpt-fast) for performant LLM inference techniques which we've adopted OOTB
- [lit-gpt](https://github.com/Lightning-AI/litgpt), [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), [transformers](https://github.com/huggingface/transformers) and [llama recipes](https://github.com/meta-llama/llama-recipes) for reference implementations and pushing forward the LLM finetuning community
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

&nbsp;

## Contributing

We welcome any feature requests, bug reports, or pull requests from the community. See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

&nbsp;

## License

TorchTune is released under the [BSD 3 license](./LICENSE). However you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models.
