

# torchtune

[![Unit Test](https://github.com/pytorch/torchtune/actions/workflows/unit_test.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtune/actions/workflows/unit_test.yaml)
![Integration Tests](https://github.com/pytorch/torchtune/actions/workflows/gpu_test.yaml/badge.svg)
[![](https://dcbadge.vercel.app/api/server/4Xsdn8Rr9Q?style=flat)](https://discord.gg/4Xsdn8Rr9Q)

[**Overview**](#overview-) | [**Installation**](#installation-%EF%B8%8F) | [**Get Started**](#get-started-) |  [**Documentation**](https://pytorch.org/torchtune/main/index.html) | [**Community**](#community-) | [**Citing torchtune**](#citing-torchtune-) | [**License**](#license)

### 📣 Recent updates 📣
* *May 2025*: torchtune has added support for *Qwen3* models! Check out all the configs [here](recipes/configs/qwen3)
* *April 2025*: **Llama4** is now available in torchtune! Try out our full and LoRA finetuning configs [here](recipes/configs/llama4)
* *February 2025*: Multi-node training is officially [open for business in torchtune](https://pytorch.org/torchtune/main/tutorials/multinode.html)! Full finetune on multiple nodes to take advantage of larger batch sizes and models.
* *December 2024*: torchtune now supports **Llama 3.3 70B**! Try it out by following our installation instructions [here](#installation-%EF%B8%8F), then run any of the configs [here](recipes/configs/llama3_3).
* *November 2024*: torchtune has released [v0.4.0](https://github.com/pytorch/torchtune/releases/tag/v0.4.0) which includes stable support for exciting features like activation offloading and multimodal QLoRA
* *November 2024*: torchtune has added [Gemma2](recipes/configs/gemma2) to its models!
* *October 2024*: torchtune added support for Qwen2.5 models - find the configs [here](recipes/configs/qwen2_5/)
* *September 2024*: torchtune has support for **Llama 3.2 11B Vision**, **Llama 3.2 3B**, and **Llama 3.2 1B** models! Try them out by following our installation instructions [here](#installation-%EF%B8%8F), then run any of the text configs [here](recipes/configs/llama3_2) or vision configs [here](recipes/configs/llama3_2_vision).


&nbsp;

## Overview 📚


torchtune is a PyTorch library for easily authoring, post-training, and experimenting with LLMs. It provides:

- Hackable training recipes for SFT, knowledge distillation, DPO, PPO, GRPO, and quantization-aware training
- Simple PyTorch implementations of popular LLMs like Llama, Gemma, Mistral, Phi, Qwen, and more
- Best-in-class memory efficiency, performance improvements, and scaling, utilizing the latest PyTorch APIs
- YAML configs for easily configuring training, evaluation, quantization or inference recipes

&nbsp;

### Post-training recipes

torchtune supports [the entire post-training lifecycle](https://pytorch.org/torchtune/main/recipes/recipes_overview.html). A successful post-trained model will likely utilize several of the below methods.

#### Supervised Finetuning (SFT)

| Type of Weight Update | 1 Device | >1 Device | >1 Node |
|-----------------------|:--------:|:---------:|:-------:|
| Full                  |    ✅    |     ✅    |   ✅    |
| [LoRA/QLoRA](https://pytorch.org/torchtune/stable/recipes/lora_finetune_single_device.html)            |    ✅    |     ✅    |    ✅    |

Example: ``tune run lora_finetune_single_device --config llama3_2/3B_lora_single_device`` <br />
You can also run e.g. ``tune ls lora_finetune_single_device`` for a full list of available configs.

#### [Knowledge Distillation (KD)](https://pytorch.org/torchtune/0.4/tutorials/llama_kd_tutorial.html)

| Type of Weight Update | 1 Device | >1 Device | >1 Node |
|-----------------------|:--------:|:---------:|:-------:|
| Full                  |    ❌    |     ❌    |    ❌    |
| LoRA/QLoRA            |    ✅    |     ✅    |    ❌    |

Example: ``tune run knowledge_distillation_distributed --config qwen2/1.5B_to_0.5B_KD_lora_distributed`` <br />
You can also run e.g. ``tune ls knowledge_distillation_distributed`` for a full list of available configs.

#### Reinforcement Learning / Reinforcement Learning from Human Feedback (RLHF)

| Method | Type of Weight Update | 1 Device | >1 Device | >1 Node |
|------------------------------|-----------------------|:--------:|:---------:|:-------:|
| [DPO](https://pytorch.org/torchtune/stable/recipes/dpo.html)                          | Full                  |    ❌    |     ✅    |    ❌    |
|                           | LoRA/QLoRA            |    ✅    |     ✅    |    ❌    |
| PPO                          | Full                  |    ✅    |     ❌    |    ❌    |
|                           | LoRA/QLoRA            |    ❌    |     ❌    |    ❌    |
| GRPO                         | Full                  |    🚧    |     ✅    |  ✅   |
|                           | LoRA/QLoRA            |    ❌    |     ❌    |    ❌    |

Example: ``tune run lora_dpo_single_device --config llama3_1/8B_dpo_single_device`` <br />
You can also run e.g. ``tune ls full_dpo_distributed`` for a full list of available configs.

#### [Quantization-Aware Training (QAT)](https://pytorch.org/torchtune/main/tutorials/qat_finetune.html)

| Type of Weight Update | 1 Device | >1 Device | >1 Node |
|-----------------------|:--------:|:---------:|:-------:|
| [Full](https://pytorch.org/torchtune/stable/recipes/qat_distributed.html)                  |    ✅     |     ✅    |    ❌    |
| LoRA/QLoRA            |    ❌    |     ✅    |    ❌    |

Example: ``tune run qat_distributed --config llama3_1/8B_qat_lora`` <br />
You can also run e.g. ``tune ls qat_distributed`` or ``tune ls qat_single_device`` for a full list of available configs.

The above configs are just examples to get you started. The full list of recipes can be found [here](recipes/). If you'd like to work on one of the gaps you see, please submit a PR! If there's a entirely new post-training method you'd like to see implemented in torchtune, feel free to open an Issue.

&nbsp;

### Models

For the above recipes, torchtune supports many state-of-the-art models available on the [Hugging Face Hub](https://huggingface.co/models) or [Kaggle Hub](https://www.kaggle.com/models). Some of our supported models:

| Model                                         | Sizes     |
|-----------------------------------------------|-----------|
| [Llama4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4)    | Scout (17B x 16E) [[models](torchtune/models/llama4/_model_builders.py), [configs](recipes/configs/llama4/)]        |
| [Llama3.3](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_3)    | 70B [[models](torchtune/models/llama3_3/_model_builders.py), [configs](recipes/configs/llama3_3/)]        |
| [Llama3.2-Vision](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2#-llama-3.2-vision-models-(11b/90b)-)    | 11B, 90B [[models](torchtune/models/llama3_2_vision/_model_builders.py), [configs](recipes/configs/llama3_2_vision/)]        |
| [Llama3.2](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2)    | 1B, 3B [[models](torchtune/models/llama3_2/_model_builders.py), [configs](recipes/configs/llama3_2/)]        |
| [Llama3.1](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1)    | 8B, 70B, 405B [[models](torchtune/models/llama3_1/_model_builders.py), [configs](recipes/configs/llama3_1/)]        |
| [Mistral](https://huggingface.co/mistralai)   | 7B [[models](torchtune/models/mistral/_model_builders.py), [configs](recipes/configs/mistral/)] |
| [Gemma2](https://huggingface.co/docs/transformers/main/en/model_doc/gemma2)   | 2B, 9B, 27B [[models](torchtune/models/gemma2/_model_builders.py), [configs](recipes/configs/gemma2/)] |
| [Microsoft Phi4](https://huggingface.co/collections/microsoft/phi-4-677e9380e514feb5577a40e4) | 14B [[models](torchtune/models/phi4/), [configs](recipes/configs/phi4/)]
| [Microsoft Phi3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) | Mini [[models](torchtune/models/phi3/), [configs](recipes/configs/phi3/)]
| [Qwen3](https://qwenlm.github.io/blog/qwen3/) | 0.6B, 1.7B, 4B, 8B, 14B, 32B [[models](torchtune/models/qwen3/), [configs](recipes/configs/qwen3/)]
| [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/) | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B [[models](torchtune/models/qwen2_5/), [configs](recipes/configs/qwen2_5/)]
| [Qwen2](https://qwenlm.github.io/blog/qwen2/) | 0.5B, 1.5B, 7B [[models](torchtune/models/qwen2/), [configs](recipes/configs/qwen2/)]

We're always adding new models, but feel free to [file an issue](https://github.com/pytorch/torchtune/issues/new) if there's a new one you would like to see in torchtune.

&nbsp;

### Memory and training speed

Below is an example of the memory requirements and training speed for different Llama 3.1 models.

> [!NOTE]
> For ease of comparison, all the below numbers are provided for batch size 2 (without gradient accumulation), a dataset packed to sequence length 2048, and torch compile enabled.

If you are interested in running on different hardware or with different models, check out our documentation on memory optimizations [here](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html) to find the right setup for you.

| Model | Finetuning Method | Runnable On | Peak Memory per GPU | Tokens/sec * |
|:-:|:-:|:-:|:-:|:-:|
| Llama 3.1 8B | Full finetune | 1x 4090 | 18.9 GiB | 1650 |
| Llama 3.1 8B | Full finetune | 1x A6000 | 37.4 GiB |  2579|
| Llama 3.1 8B | LoRA | 1x 4090 |  16.2 GiB | 3083 |
| Llama 3.1 8B | LoRA | 1x A6000 | 30.3 GiB  | 4699 |
| Llama 3.1 8B | QLoRA | 1x 4090 | 7.4 GiB | 2413  |
| Llama 3.1 70B | Full finetune | 8x A100  | 13.9 GiB ** | 1568  |
| Llama 3.1 70B | LoRA | 8x A100 | 27.6 GiB  | 3497  |
| Llama 3.1 405B | QLoRA | 8x A100 | 44.8 GB  | 653  |

*= Measured over one full training epoch <br />
**= Uses CPU offload with fused optimizer

&nbsp;

### Optimization flags

torchtune exposes a number of levers for memory efficiency and performance. The table below demonstrates the effects of applying some of these techniques sequentially to the Llama 3.2 3B model. Each technique is added on top of the previous one, except for LoRA and QLoRA, which do not use `optimizer_in_bwd` or `AdamW8bit` optimizer.

> Baseline uses Recipe=**full_finetune_single_device**, Model=**Llama 3.2 3B**, Batch size=**2**, Max sequence length=**4096**, Precision=**bf16**, Hardware=**A100**

| Technique | Peak Memory Active (GiB) | % Change Memory vs Previous | Tokens Per Second | % Change Tokens/sec vs Previous|
|:--|:-:|:-:|:-:|:-:|
| Baseline | 25.5 | - | 2091 | - |
| [+ Packed Dataset](https://pytorch.org/torchtune/main/basics/packing.html) | 60.0 | +135.16% | 7075 | +238.40% |
| [+ Compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) | 51.0 | -14.93% | 8998 | +27.18% |
| [+ Chunked Cross Entropy](https://pytorch.org/torchtune/main/generated/torchtune.modules.loss.CEWithChunkedOutputLoss.html) | 42.9 | -15.83% | 9174 | +1.96% |
| [+ Activation Checkpointing](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html#activation-checkpointing) | 24.9 | -41.93% | 7210 | -21.41% |
| [+ Fuse optimizer step into backward](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html#fusing-optimizer-step-into-backward-pass) | 23.1 | -7.29% | 7309 | +1.38% |
| [+ Activation Offloading](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html#activation-offloading) | 21.8 | -5.48% | 7301 | -0.11% |
| [+ 8-bit AdamW](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html#lower-precision-optimizers) | 17.6 | -19.63% | 6960 | -4.67% |
| [LoRA](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html#glossary-lora) | 8.5 | -51.61% | 8210 | +17.96% |
| [QLoRA](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html#quantized-low-rank-adaptation-qlora) | 4.6 | -45.71% | 8035 | -2.13% |

The final row in the table vs baseline + Packed Dataset uses **81.9%** less memory with a **284.3%** increase in tokens per second.

<details>
<summary>Command to reproduce final row.</summary>

```bash
tune run lora_finetune_single_device --config llama3_2/3B_qlora_single_device \
dataset.packed=True \
compile=True \
loss=torchtune.modules.loss.CEWithChunkedOutputLoss \
enable_activation_checkpointing=True \
optimizer_in_bwd=False \
enable_activation_offloading=True \
optimizer=torch.optim.AdamW \
tokenizer.max_seq_len=4096 \
gradient_accumulation_steps=1 \
epochs=1 \
batch_size=2
```

</details>

&nbsp;

## Installation 🛠️


torchtune is **only** tested with the latest stable PyTorch release (currently 2.6.0) as well as the preview nightly version, and leverages
torchvision for finetuning multimodal LLMs and torchao for the latest in quantization techniques; you should install these as well.

### Install stable release

```bash
# Install stable PyTorch, torchvision, torchao stable releases
pip install torch torchvision torchao
pip install torchtune
```

### Install nightly release

```bash
# Install PyTorch, torchvision, torchao nightlies.
pip install --pre --upgrade torch torchvision torchao --index-url https://download.pytorch.org/whl/nightly/cu126 # full options are cpu/cu118/cu124/cu126/xpu/rocm6.2/rocm6.3/rocm6.4
pip install --pre --upgrade torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

You can also check out our [install documentation](https://pytorch.org/torchtune/main/install.html) for more information, including installing torchtune from source.

&nbsp;

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

&nbsp;

## Get Started 🚀


To get started with torchtune, see our [First Finetune Tutorial](https://pytorch.org/torchtune/main/tutorials/first_finetune_tutorial.html). Our [End-to-End Workflow Tutorial](https://pytorch.org/torchtune/main/tutorials/e2e_flow.html) will show you how to evaluate, quantize, and run inference with a Llama model. The rest of this section will provide a quick overview of these steps with Llama3.1.


### Downloading a model

Follow the instructions on the official [`meta-llama`](https://huggingface.co/meta-llama) repository to ensure you have access to the official Llama model weights. Once you have confirmed access, you can run the following command to download the weights to your local machine. This will also download the tokenizer model and a responsible use guide.

To download Llama3.1, you can run:

```bash
tune download meta-llama/Meta-Llama-3.1-8B-Instruct \
--output-dir /tmp/Meta-Llama-3.1-8B-Instruct \
--ignore-patterns "original/consolidated.00.pth" \
--hf-token <HF_TOKEN> \
```

> [!Tip]
> Set your environment variable `HF_TOKEN` or pass in `--hf-token` to the command in order to validate your access. You can find your token at https://huggingface.co/settings/tokens

### Running finetuning recipes

You can finetune Llama3.1 8B with LoRA on a single GPU using the following command:

```bash
tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device
```

For distributed training, tune CLI integrates with [torchrun](https://pytorch.org/docs/stable/elastic/run.html).
To run a full finetune of Llama3.1 8B on two GPUs:

```bash
tune run --nproc_per_node 2 full_finetune_distributed --config llama3_1/8B_full
```

> [!Tip]
> Make sure to place any torchrun commands **before** the recipe specification. Any CLI args after this will override the config and not impact distributed training.

### Modify Configs

There are two ways in which you can modify configs:

**Config Overrides**

You can directly overwrite config fields from the command line:

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
tune cp llama3_1/8B_full ./my_custom_config.yaml
Copied to ./my_custom_config.yaml
```

Then, you can run your custom recipe by directing the `tune run` command to your local files:

```bash
tune run full_finetune_distributed --config ./my_custom_config.yaml
```

Check out `tune --help` for all possible CLI commands and options. For more information on using and updating configs, take a look at our [config deep-dive](https://pytorch.org/torchtune/main/deep_dives/configs.html).

### Custom Datasets

torchtune supports finetuning on a variety of different datasets, including [instruct-style](https://pytorch.org/torchtune/main/basics/instruct_datasets.html), [chat-style](https://pytorch.org/torchtune/main/basics/chat_datasets.html), [preference datasets](https://pytorch.org/torchtune/main/basics/preference_datasets.html), and more. If you want to learn more about how to apply these components to finetune on your own custom dataset, please check out the provided links along with our [API docs](https://pytorch.org/torchtune/main/api_ref_datasets.html).

### Custom Devices
torchtune supports finetuning on a variety of devices, including NVIDIA GPU, Intel XPU, AMD ROCm, Apple MPS, and Ascend NPU. If you're interested in running recipes on a custom device, such as Intel XPU, follow the steps below.

Step 1: Refer to the [Getting Started on Intel GPU guide](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) to configure your environment.

Step 2: Update device information via either CLI override or config changes.
You can directly overwrite config fields from the command line:

```bash
tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device device=xpu
```
Or edit your local copy of configuration files and replace `device: cuda` with `device: xpu`

&nbsp;

## Community 🌍

torchtune focuses on integrating with popular tools and libraries from the ecosystem. These are just a few examples, with more under development:

- [Hugging Face Hub](https://huggingface.co/docs/hub/en/index) for [accessing model weights](torchtune/_cli/download.py)
- [EleutherAI's LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness) for [evaluating](recipes/eleuther_eval.py) trained models
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/en/index) for [access](torchtune/datasets/_instruct.py) to training and evaluation datasets
- [PyTorch FSDP2](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md) for distributed training
- [torchao](https://github.com/pytorch/ao) for lower precision dtypes and [post-training quantization](recipes/quantize.py) techniques
- [Weights & Biases](https://wandb.ai/site) for [logging](https://pytorch.org/torchtune/main/deep_dives/wandb_logging.html) metrics and checkpoints, and tracking training progress
- [Comet](https://www.comet.com/site/) as another option for [logging](https://pytorch.org/torchtune/main/deep_dives/comet_logging.html)
- [ExecuTorch](https://pytorch.org/executorch-overview) for [on-device inference](https://github.com/pytorch/executorch/tree/main/examples/models/llama2#optional-finetuning) using finetuned models
- [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index) for low memory optimizers for our [single-device recipes](recipes/configs/llama2/7B_full_low_memory.yaml)
- [PEFT](https://github.com/huggingface/peft) for continued finetuning or inference with torchtune models in the Hugging Face ecosystem

&nbsp;

### Community Contributions

We really value our community and the contributions made by our wonderful users. We'll use this section to call out some of these contributions. If you'd like to help out as well, please see the [CONTRIBUTING](CONTRIBUTING.md) guide.

- [@SalmanMohammadi](https://github.com/salmanmohammadi) for adding a comprehensive end-to-end recipe for [Reinforcement Learning from Human Feedback (RLHF)](recipes/ppo_full_finetune_single_device.py) finetuning with PPO to torchtune
- [@fyabc](https://github.com/fyabc) for adding Qwen2 models, tokenizer, and recipe integration to torchtune
- [@solitude-alive](https://github.com/solitude-alive) for adding the [Gemma 2B model](torchtune/models/gemma/) to torchtune, including recipe changes, numeric validations of the models and recipe correctness
- [@yechenzhi](https://github.com/yechenzhi) for adding [Direct Preference Optimization (DPO)](recipes/lora_dpo_single_device.py) to torchtune, including the recipe and config along with correctness checks
- [@Optimox](https://github.com/Optimox) for adding all the [Gemma2 variants](torchtune/models/gemma2) to torchtune!


&nbsp;

## Acknowledgements 🙏

The transformer code in this repository is inspired by the original [Llama2 code](https://github.com/meta-llama/llama/blob/main/llama/model.py). We also want to give a huge shout-out to EleutherAI, Hugging Face and
Weights & Biases for being wonderful collaborators and for working with us on some of these integrations within torchtune. In addition, we want to acknowledge some other awesome libraries and tools from the ecosystem:

- [gpt-fast](https://github.com/meta-pytorch/gpt-fast) for performant LLM inference techniques which we've adopted out-of-the-box
- [llama recipes](https://github.com/meta-llama/llama-recipes) for spring-boarding the llama2 community
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for bringing several memory and performance based techniques to the PyTorch ecosystem
- [@winglian](https://github.com/winglian/) and [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) for early feedback and brainstorming on torchtune's design and feature set.
- [lit-gpt](https://github.com/Lightning-AI/litgpt) for pushing the LLM finetuning community forward.
- [HF TRL](https://github.com/huggingface/trl) for making reward modeling more accessible to the PyTorch community.

&nbsp;

## Citing torchtune 📝

If you find the torchtune library useful, please cite it in your work as below.

```bibtex
@software{torchtune,
  title = {torchtune: PyTorch's finetuning library},
  author = {torchtune maintainers and contributors},
  url = {https//github.com/pytorch/torchtune},
  license = {BSD-3-Clause},
  month = apr,
  year = {2024}
}
```

&nbsp;

## License

torchtune is released under the [BSD 3 license](./LICENSE). However you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models.
