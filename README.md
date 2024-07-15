
[![Unit Test](https://github.com/pytorch/torchtune/actions/workflows/unit_test.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtune/actions/workflows/unit_test.yaml)
![Recipe Integration Test](https://github.com/pytorch/torchtune/actions/workflows/recipe_test.yaml/badge.svg)
[![](https://dcbadge.vercel.app/api/server/4Xsdn8Rr9Q?style=flat)](https://discord.gg/4Xsdn8Rr9Q)

&nbsp;
&nbsp;

torchtune now officially supports Meta Llama3! Check out our recipes for Llama3-8B-Instruct with LoRA, QLoRA and Full fine-tune in the [Llama3](#llama3) section! We also support 70B fine-tuning with LoRA! 🚀 🦙

# torchtune

[**Introduction**](#introduction) | [**Installation**](#installation) | [**Get Started**](#get-started) |  [**Documentation**](https://pytorch.org/torchtune/main/index.html) | [**Design Principles**](#design-principles) | [**Community Contributions**](#community-contributions) | [**License**](#license)

&nbsp;

## Introduction

torchtune is a PyTorch-native library for easily authoring, fine-tuning and experimenting with LLMs. We're excited to announce our alpha release!

torchtune provides:

- Native-PyTorch implementations of popular LLMs using composable and modular building blocks
- Easy-to-use and hackable training recipes for popular fine-tuning techniques (LoRA, QLoRA) - no trainers, no frameworks, just PyTorch!
- YAML configs for easily configuring training, evaluation, quantization or inference recipes
- Built-in support for many popular dataset formats and prompt templates to help you quickly get started with training

torchtune focuses on integrating with popular tools and libraries from the ecosystem. These are just a few examples, with more under development:

- [Hugging Face Hub](https://huggingface.co/docs/hub/en/index) for [accessing model weights](torchtune/_cli/download.py)
- [EleutherAI's LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness) for [evaluating](recipes/eleuther_eval.py) trained models
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/en/index) for [access](torchtune/datasets/_instruct.py) to training and evaluation datasets
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) for distributed training
- [torchao](https://github.com/pytorch-labs/ao) for lower precision dtypes and [post-training quantization](recipes/quantize.py) techniques
- [Weights & Biases](https://wandb.ai/site) for [logging](https://pytorch.org/torchtune/main/deep_dives/wandb_logging.html) metrics and checkpoints, and tracking training progress
- [ExecuTorch](https://pytorch.org/executorch-overview) for [on-device inference](https://github.com/pytorch/executorch/tree/main/examples/models/llama2#optional-finetuning) using fine-tuned models
- [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index) for low memory optimizers for our [single-device recipes](recipes/configs/llama2/7B_full_low_memory.yaml)

&nbsp;

### Models

torchtune currently supports the following models.

| Model                                         | Sizes     |
|-----------------------------------------------|-----------|
| [Llama3](https://llama.meta.com/llama3)    | 8B, 70B [[models](torchtune/models/llama3/_model_builders.py), [configs](recipes/configs/llama3/)]        |
| [Llama2](https://llama.meta.com/llama2/)   | 7B, 13B, 70B [[models](torchtune/models/llama2/_model_builders.py), [configs](recipes/configs/llama2/)]        |
| [Code-Llama2](https://ai.meta.com/blog/code-llama-large-language-model-coding/)   | 7B, 13B, 70B [[model](torchtune/models/code_llama2/_model_builders.py), [configs](recipes/configs/code_llama2/)] |
| [Mistral](https://huggingface.co/mistralai)   | 7B [[model](torchtune/models/mistral/_model_builders.py), [configs](recipes/configs/mistral/)] |
| [Gemma](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b)   | 2B, 7B [[model](torchtune/models/gemma/_model_builders.py), [configs](recipes/configs/gemma/)] |
| [Microsoft Phi3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) | Mini [[model](torchtune/models/phi3/), [configs](recipes/configs/phi3/)]

We'll be adding a number of new models in the coming weeks, including support for 70B versions and MoEs.

&nbsp;

### Fine-tuning recipes

torchtune provides the following fine-tuning recipes.

| Training                           | Fine-tuning Method                 |
|------------------------------------|------------------------------------|
| Distributed Training [1 to 8 GPUs] | Full [[code](recipes/full_finetune_distributed.py), [example](recipes/configs/llama3/8B_full.yaml)], LoRA [[code](recipes/lora_finetune_distributed.py), [example](recipes/configs/llama3/8B_lora.yaml)] |
| Single Device / Low Memory [1 GPU] | Full [[code](recipes/full_finetune_single_device.py), [example](recipes/configs/llama3/8B_full_single_device.yaml)], LoRA + QLoRA [[code](recipes/lora_finetune_single_device.py), [example](recipes/configs/llama3/8B_lora_single_device.yaml)] |
| Single Device [1 GPU]              | DPO [[code](recipes/full_finetune_distributed.py), [example](recipes/configs/llama2/7B_lora_dpo_single_device.yaml)]

&nbsp;


Memory efficiency is important to us. All of our recipes are tested on a variety of setups including commodity GPUs with 24GB of VRAM as well as beefier options found in data centers.

Single-GPU recipes expose a number of memory optimizations that aren't available in the distributed versions. These include support for low-precision optimizers from [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index) and fusing optimizer step with backward to reduce memory footprint from the gradients (see example [config](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full_low_memory.yaml)). For memory-constrained setups, we recommend using the single-device configs as a starting point.

This table captures the peak memory usage and training speed for recipes in torchtune.

| Example HW Resources | Finetuning Method | Model   | Setting    | Peak Memory per GPU (GB) | Training Speed (tokens/sec) |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 x RTX 4090 |     QLoRA **         |  Llama2-7B      |    Batch Size = 4, Seq Length = 2048   | 12.3 GB | 3155  |
| 1 x RTX 4090 |     LoRA          |  Llama2-7B      |    Batch Size = 4, Seq Length = 2048   | 21.3 GB | 2582  |
| 2 x RTX 4090 |     LoRA          |  Llama2-7B      |    Batch Size = 4, Seq Length = 2048   | 16.2 GB | 2768  |
| 1 x RTX 4090 |   Full finetune *   |  Llama2-7B      |    Batch Size = 4, Seq Length = 2048   | 24.1 GB | 702  |
| 4 x RTX 4090 |   Full finetune   |  Llama2-7B      |    Batch Size = 4, Seq Length = 2048   | 24.1 GB | 1388  |
| 8 x A100     |     LoRA          |  Llama2-70B     |    Batch Size = 4, Seq Length = 4096   | 26.4 GB | 3384  |
| 8 x A100     |   Full Finetune *   |  Llama2-70B     |    Batch Size = 4, Seq Length = 4096   | 70.4 GB | 2032  |

*= Uses PagedAdamW from bitsandbytes

**= Uses torch compile


&nbsp;

## Llama3

torchtune supports fine-tuning for the Llama3 8B and 70B size models. We currently support LoRA, QLoRA and full fine-tune on a single GPU as well as LoRA and full fine-tune on multiple devices for the 8B model, and LoRA on multiple devices for the 70B model. For all the details, take a look at our [tutorial](https://pytorch.org/torchtune/main/tutorials/llama3.html).

**Note**: our Llama3 LoRA and QLoRA configs default to the instruct fine-tuned models.
This is because not all special token embeddings are initialized in the base 8B and 70B models.

In our initial experiments for Llama3-8B, QLoRA has a peak allocated memory of ``~9GB`` while LoRA on a single GPU has a peak allocated memory of ``~19GB``. To get started, you can use our default configs to kick off training.

### Single GPU

LoRA 8B

```bash
tune run lora_finetune_single_device --config llama3/8B_lora_single_device
```

QLoRA 8B

```bash
tune run lora_finetune_single_device --config llama3/8B_qlora_single_device
```

Full 8B

```bash
tune run full_finetune_single_device --config llama3/8B_full_single_device
```

### Multi GPU

Full 8B

```bash
tune run --nproc_per_node 4 full_finetune_distributed --config llama3/8B_full
```

LoRA 8B

```bash
tune run --nproc_per_node 2 lora_finetune_distributed --config llama3/8B_lora
```

LoRA 70B

Note that the download command for the Meta-Llama3 70B model slightly differs from download commands for the 8B models. This is because we use the HuggingFace [safetensor](https://huggingface.co/docs/safetensors/en/index) model format to load the model. To download the 70B model, run
```bash
tune download meta-llama/Meta-Llama-3-70b --hf-token <> --output-dir /tmp/Meta-Llama-3-70b --ignore-patterns "original/consolidated*"
```

Then, a finetune can be kicked off:

```bash
tune run --nproc_per_node 8 lora_finetune_distributed --config recipes/configs/llama3/70B_lora.yaml
```

You can find a full list of all our Llama3 configs [here.](recipes/configs/llama3)


&nbsp;

---

## Installation

**Step 1:** [Install PyTorch](https://pytorch.org/get-started/locally/). torchtune is tested with the latest stable PyTorch release as well as the preview nightly version. For fine-tuning the multimodal LLMs available in the repo, you'll need to install torchvision as well.

```
# Install stable version of PyTorch using pip
pip install torch torchvision

# Nightly install for latest features
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121
```

**Step 2:** The latest stable version of torchtune is hosted on PyPI and can be downloaded with the following command:

```bash
pip install torchtune
```

To confirm that the package is installed correctly, you can run the following command:

```bash
tune --help
```

And should see the following output:

```bash
usage: tune [-h] {ls,cp,download,run,validate} ...

Welcome to the torchtune CLI!

options:
  -h, --help            show this help message and exit

...
```

You can also install the latest and greatest torchtune has to offer by [installing a nightly build](https://pytorch.org/torchtune/main/install.html).

&nbsp;

---

## Get Started

To get started with fine-tuning your first LLM with torchtune, see our tutorial on [fine-tuning Llama2 7B](https://pytorch.org/torchtune/main/tutorials/first_finetune_tutorial.html). Our [end-to-end workflow](https://pytorch.org/torchtune/main/tutorials/e2e_flow.html) tutorial will show you how to evaluate, quantize and run inference with this model. The rest of this section will provide a quick overview of these steps with Llama2.

&nbsp;

### Downloading a model

Follow the instructions on the official [`meta-llama`](https://huggingface.co/meta-llama) repository to ensure you have access to the official Llama model weights. Once you have confirmed access, you can run the following command to download the weights to your local machine. This will also download the tokenizer model and a responsible use guide.

### Llama2 download
```bash
tune download meta-llama/Llama-2-7b-hf \
--output-dir /tmp/Llama-2-7b-hf \
--hf-token <HF_TOKEN> \
```

### Llama3 download
```bash
tune download meta-llama/Meta-Llama-3-8B \
--output-dir /tmp/Meta-Llama-3-8B \
--hf-token <HF_TOKEN> \
```


> Tip: Set your environment variable `HF_TOKEN` or pass in `--hf-token` to the command in order to validate your access.
You can find your token at https://huggingface.co/settings/tokens

&nbsp;

### Running fine-tuning recipes

Llama2 7B + LoRA on single GPU:

```bash
tune run lora_finetune_single_device --config llama2/7B_lora_single_device
```

For distributed training, tune CLI integrates with [torchrun](https://pytorch.org/docs/stable/elastic/run.html).
Llama2 7B + LoRA on two GPUs:

```bash
tune run --nproc_per_node 2 full_finetune_distributed --config llama2/7B_full
```

> Tip: Make sure to place any torchrun commands **before** the recipe specification. Any CLI args after this will override the config and not impact distributed training.

&nbsp;

### Modify Configs

There are two ways in which you can modify configs:

**Config Overrides**

You can easily overwrite config properties from the command-line:

```bash
tune run lora_finetune_single_device \
--config llama2/7B_lora_single_device \
batch_size=8 \
enable_activation_checkpointing=True \
max_steps_per_epoch=128
```

**Update a Local Copy**

You can also copy the config to your local directory and modify the contents directly:

```bash
tune cp llama2/7B_full ./my_custom_config.yaml
Copied to ./7B_full.yaml
```

Then, you can run your custom recipe by directing the `tune run` command to your local files:

```bash
tune run full_finetune_distributed --config ./my_custom_config.yaml
```

&nbsp;

Check out `tune --help` for all possible CLI commands and options. For more information on using and updating configs, take a look at our [config deep-dive](https://pytorch.org/torchtune/main/deep_dives/configs.html).

&nbsp;

## Design Principles

torchtune embodies PyTorch’s design philosophy [[details](https://pytorch.org/docs/stable/community/design.html)], especially "usability over everything else".

### Native PyTorch

torchtune is a native-PyTorch library. While we provide integrations with the surrounding ecosystem (eg: Hugging Face Datasets, EleutherAI Eval Harness), all of the core functionality is written in PyTorch.

### Simplicity and Extensibility

torchtune is designed to be easy to understand, use and extend.

- Composition over implementation inheritance - layers of inheritance for code re-use makes the code hard to read and extend
- No training frameworks - explicitly outlining the training logic makes it easy to extend for custom use cases
- Code duplication is preferred over unnecessary abstractions
- Modular building blocks over monolithic components

### Correctness

torchtune provides well-tested components with a high-bar on correctness. The library will never be the first to provide a feature, but available features will be thoroughly tested. We provide

- Extensive unit-tests to ensure component-level numerical parity with reference implementations
- Checkpoint-tests to ensure model-level numerical parity with reference implementations
- Integration tests to ensure recipe-level performance parity with reference implementations on standard benchmarks

&nbsp;

## Community Contributions

We really value our community and the contributions made by our wonderful users. We'll use this section to call out some of these contributions! If you'd like to help out as well, please see the [CONTRIBUTING](CONTRIBUTING.md) guide.

- [@solitude-alive](https://github.com/solitude-alive) for adding the [Gemma 2B model](torchtune/models/gemma/) to torchtune, including recipe changes, numeric validations of the models and recipe correctness
- [@yechenzhi](https://github.com/yechenzhi) for adding [DPO](recipes/lora_dpo_single_device.py) to torchtune, including the recipe and config along with correctness checks


&nbsp;

## Acknowledgements

The Llama2 code in this repository is inspired by the original [Llama2 code](https://github.com/meta-llama/llama/blob/main/llama/model.py).

We want to give a huge shout-out to EleutherAI, Hugging Face and Weights & Biases for being wonderful collaborators and for working with us on some of these integrations within torchtune.

We also want to acknowledge some awesome libraries and tools from the ecosystem:
- [gpt-fast](https://github.com/pytorch-labs/gpt-fast) for performant LLM inference techniques which we've adopted OOTB
- [llama recipes](https://github.com/meta-llama/llama-recipes) for spring-boarding the llama2 community
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for bringing several memory and performance based techniques to the PyTorch ecosystem
- [@winglian](https://github.com/winglian/) and [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) for early feedback and brainstorming on torchtune's design and feature set.
- [lit-gpt](https://github.com/Lightning-AI/litgpt) for pushing the LLM fine-tuning community forward.
- [HF TRL](https://github.com/huggingface/trl) for making reward modeling more accessible to the PyTorch community.

&nbsp;


## License

torchtune is released under the [BSD 3 license](./LICENSE). However you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models.
