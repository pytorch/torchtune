> ⚠️ **Torchtune 维护状态说明：** torchtune 的官方核心开发已于 2025 年逐步收尾——详情参见 [The future of torchtune](https://github.com/meta-pytorch/torchtune/issues/2883)。由衷感谢 150 多位开源贡献者共同打造了这一优秀的代码库。

# torchtune

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

[![Unit Test](https://github.com/pytorch/torchtune/actions/workflows/unit_test.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtune/actions/workflows/unit_test.yaml)
![Integration Tests](https://github.com/pytorch/torchtune/actions/workflows/gpu_test.yaml/badge.svg)
[![](https://dcbadge.vercel.app/api/server/4Xsdn8Rr9Q?style=flat)](https://discord.gg/4Xsdn8Rr9Q)

[**核心概览**](#核心概览-) | [**安装指南**](#安装指南-️) | [**快速上手**](#快速上手-) | [**官方文档**](https://pytorch.org/torchtune/main/index.html) | [**生态社区**](#生态社区-) | [**项目引用**](#项目引用-) | [**开源许可**](#开源许可)

### 📣 最新动态 📣
* *2025年5月*：torchtune 新增对 *Qwen3* 系列模型的全面支持！查阅全部配置请见[此处](recipes/configs/qwen3)
* *2025年4月*：**Llama4** 正式上线 torchtune！前往[此处](recipes/configs/llama4)体验全量微调与 LoRA 微调配置
* *2025年2月*：多节点分布式训练正式[全面就绪](https://pytorch.org/torchtune/main/tutorials/multinode.html)！支持跨多机多卡执行全量微调，从容应对超大模型与大 Batch Size 训练。
* *2024年12月*：torchtune 现已支持 **Llama 3.3 70B**！参考[安装说明](#安装指南-️)完成环境配置后，即可直接运行[此处](recipes/configs/llama3_3)的任意训练配置。
* *2024年11月*：发布 [v0.4.0](https://github.com/pytorch/torchtune/releases/tag/v0.4.0) 版本，正式稳定支持激活值卸载（Activation Offloading）与多模态 QLoRA 等特性。
* *2024年11月*：新增 [Gemma2](recipes/configs/gemma2) 系列模型支持！
* *2024年10月*：新增 Qwen2.5 系列模型支持，相关配置文件位于[此处](recipes/configs/qwen2_5/)。
* *2024年9月*：新增对 **Llama 3.2 11B Vision**、**Llama 3.2 3B** 以及 **Llama 3.2 1B** 模型的支持！参考[安装说明](#安装指南-️)体验文本模型配置（见[此处](recipes/configs/llama3_2)）与视觉模型配置（见[此处](recipes/configs/llama3_2_vision)）。

&nbsp;

## 核心概览 📚

torchtune 是一个专注于大语言模型（LLM）微调、后训练（Post-Training）与实验探索的原生 PyTorch 算法库。它具有以下特性：

- **易于定制（Hackable）的训练 Recipe**：覆盖监督微调（SFT）、知识蒸馏（KD）、直接偏好优化（DPO）、PPO、GRPO 以及量化感知训练（QAT）
- **简洁纯粹的 PyTorch 实现**：原生实现热门主流开源模型，如 Llama、Gemma、Mistral、Phi、Qwen 等
- **极致的显存效率与性能扩展**：充分释放最新 PyTorch 官方 API 的加速潜力
- **清晰易懂的 YAML 配置**：轻松编排训练、评估、模型量化或推理流程

&nbsp;

### 后训练（Post-Training）配方全景

torchtune 覆盖了[完整的模型后训练生命周期](https://pytorch.org/torchtune/main/recipes/recipes_overview.html)。一个成功的后训练流程通常结合以下多种方法：

#### 监督微调 (SFT)

| 权重更新方式 | 单卡 (1 Device) | 多卡 (>1 Device) | 多节点 (>1 Node) |
|-----------------------|:--------:|:---------:|:-------:|
| 全量微调 (Full)       |    ✅    |     ✅    |   ✅    |
| [LoRA/QLoRA](https://pytorch.org/torchtune/stable/recipes/lora_finetune_single_device.html)            |    ✅    |     ✅    |    ✅    |

运行范例：``tune run lora_finetune_single_device --config llama3_2/3B_lora_single_device`` <br />
您也可以执行 ``tune ls lora_finetune_single_device`` 查看所有预置可用配置。

#### [知识蒸馏 (Knowledge Distillation, KD)](https://pytorch.org/torchtune/0.4/tutorials/llama_kd_tutorial.html)

| 权重更新方式 | 单卡 (1 Device) | 多卡 (>1 Device) | 多节点 (>1 Node) |
|-----------------------|:--------:|:---------:|:-------:|
| 全量微调 (Full)       |    ❌    |     ❌    |    ❌    |
| LoRA/QLoRA            |    ✅    |     ✅    |    ❌    |

运行范例：``tune run knowledge_distillation_distributed --config qwen2/1.5B_to_0.5B_KD_lora_distributed`` <br />
您也可以执行 ``tune ls knowledge_distillation_distributed`` 列出所有蒸馏配置文件。

#### 强化学习 / 人类反馈强化学习 (RL / RLHF)

| 算法方法 | 权重更新方式 | 单卡 (1 Device) | 多卡 (>1 Device) | 多节点 (>1 Node) |
|------------------------------|-----------------------|:--------:|:---------:|:-------:|
| [DPO](https://pytorch.org/torchtune/stable/recipes/dpo.html)                          | 全量 (Full)           |    ❌    |     ✅    |    ❌    |
|                              | LoRA/QLoRA            |    ✅    |     ✅    |    ❌    |
| PPO                          | 全量 (Full)           |    ✅    |     ❌    |    ❌    |
|                              | LoRA/QLoRA            |    ❌    |     ❌    |    ❌    |
| GRPO                         | 全量 (Full)           |    🚧    |     ✅    |  ✅   |
|                              | LoRA/QLoRA            |    ❌    |     ❌    |    ❌    |

运行范例：``tune run lora_dpo_single_device --config llama3_1/8B_dpo_single_device`` <br />
您也可以执行 ``tune ls full_dpo_distributed`` 查看所有可用配置。

#### [量化感知训练 (Quantization-Aware Training, QAT)](https://pytorch.org/torchtune/main/tutorials/qat_finetune.html)

| 权重更新方式 | 单卡 (1 Device) | 多卡 (>1 Device) | 多节点 (>1 Node) |
|-----------------------|:--------:|:---------:|:-------:|
| [全量 (Full)](https://pytorch.org/torchtune/stable/recipes/qat_distributed.html)                  |    ✅     |     ✅    |    ❌    |
| LoRA/QLoRA            |    ❌    |     ✅    |    ❌    |

运行范例：``tune run qat_distributed --config llama3_1/8B_qat_lora`` <br />
您也可以执行 ``tune ls qat_distributed`` 或 ``tune ls qat_single_device`` 查看完整配置列表。

以上配置仅供快速入门，完整 Recipe 索引请参见 [recipes/](recipes/) 目录。

&nbsp;

### 支持模型生态

torchtune 支持 [Hugging Face Hub](https://huggingface.co/models) 与 [Kaggle Hub](https://www.kaggle.com/models) 上托管的绝大多数主流前沿模型：

| 模型架构 | 尺寸版本 |
|-----------------------------------------------|-----------|
| [Llama4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4)    | Scout (17B x 16E) [[模型代码](torchtune/models/llama4/_model_builders.py), [配置](recipes/configs/llama4/)]        |
| [Llama3.3](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_3)    | 70B [[模型代码](torchtune/models/llama3_3/_model_builders.py), [配置](recipes/configs/llama3_3/)]        |
| [Llama3.2-Vision](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2#-llama-3.2-vision-models-(11b/90b)-)    | 11B, 90B [[模型代码](torchtune/models/llama3_2_vision/_model_builders.py), [配置](recipes/configs/llama3_2_vision/)]        |
| [Llama3.2](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2)    | 1B, 3B [[模型代码](torchtune/models/llama3_2/_model_builders.py), [配置](recipes/configs/llama3_2/)]        |
| [Llama3.1](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1)    | 8B, 70B, 405B [[模型代码](torchtune/models/llama3_1/_model_builders.py), [配置](recipes/configs/llama3_1/)]        |
| [Mistral](https://huggingface.co/mistralai)   | 7B [[模型代码](torchtune/models/mistral/_model_builders.py), [配置](recipes/configs/mistral/)] |
| [Gemma2](https://huggingface.co/docs/transformers/main/en/model_doc/gemma2)   | 2B, 9B, 27B [[模型代码](torchtune/models/gemma2/_model_builders.py), [配置](recipes/configs/gemma2/)] |
| [Microsoft Phi4](https://huggingface.co/collections/microsoft/phi-4-677e9380e514feb5577a40e4) | 14B [[模型代码](torchtune/models/phi4/), [配置](recipes/configs/phi4/)] |
| [Microsoft Phi3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) | Mini [[模型代码](torchtune/models/phi3/), [配置](recipes/configs/phi3/)] |
| [Qwen3](https://qwenlm.github.io/blog/qwen3/) | 0.6B, 1.7B, 4B, 8B, 14B, 32B [[模型代码](torchtune/models/qwen3/), [配置](recipes/configs/qwen3/)] |
| [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/) | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B [[模型代码](torchtune/models/qwen2_5/), [配置](recipes/configs/qwen2_5/)] |
| [Qwen2](https://qwenlm.github.io/blog/qwen2/) | 0.5B, 1.5B, 7B [[模型代码](torchtune/models/qwen2/), [配置](recipes/configs/qwen2/)] |

&nbsp;

### 显存需求与训练速度基准

下表给出了不同规模 Llama 3.1 模型的显存占用与吞吐量基准测试：

> [!NOTE]
> 为便于公平对比，下列测试均统一采用：Batch Size = 2（未启用梯度累加）、数据集打包长度（Packed Sequence Length）= 2048，并开启 `torch.compile` 编译加速。

更多跨硬件或跨模型的显存优化调优策略，可参考[官方显存优化指南](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html)。

| 模型 | 微调方法 | 可运行硬件 | 单 GPU 峰值显存 | 吞吐量 (Tokens/sec) * |
|:-:|:-:|:-:|:-:|:-:|
| Llama 3.1 8B | 全量微调 (Full) | 1x RTX 4090 | 18.9 GiB | 1650 |
| Llama 3.1 8B | 全量微调 (Full) | 1x A6000 | 37.4 GiB |  2579 |
| Llama 3.1 8B | LoRA | 1x RTX 4090 |  16.2 GiB | 3083 |
| Llama 3.1 8B | LoRA | 1x A6000 | 30.3 GiB  | 4699 |
| Llama 3.1 8B | QLoRA | 1x RTX 4090 | 7.4 GiB | 2413  |
| Llama 3.1 70B | 全量微调 (Full) | 8x A100  | 13.9 GiB ** | 1568  |
| Llama 3.1 70B | LoRA | 8x A100 | 27.6 GiB  | 3497  |
| Llama 3.1 405B | QLoRA | 8x A100 | 44.8 GB  | 653  |

\* = 在完整的一个训练 Epoch 上测量 <br />
\** = 使用 CPU 卸载与融合优化器（Fused Optimizer）

&nbsp;

### 显存与性能优化调优阶梯

torchtune 提供了丰富的性能调节旋钮。下表展示了在 Llama 3.2 3B 模型上依次应用各项优化技术带来的显存与吞吐量递进变化（除 LoRA 与 QLoRA 未使用 `optimizer_in_bwd` 或 `AdamW8bit` 外，各项技术均在上一项基础上叠加）：

> 基准设置：Recipe = **full_finetune_single_device**，Model = **Llama 3.2 3B**，Batch size = **2**，Max sequence length = **4096**，Precision = **bf16**，Hardware = **A100**

| 优化技术 | 活跃峰值显存 (GiB) | 显存相对变动率 | 吞吐量 (Tokens/sec) | 吞吐量相对变动率 |
|:--|:-:|:-:|:-:|:-:|
| Baseline（基线） | 25.5 | - | 2091 | - |
| [+ Packed Dataset（数据打包）](https://pytorch.org/torchtune/main/basics/packing.html) | 60.0 | +135.16% | 7075 | +238.40% |
| [+ Compile（图编译优化）](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) | 51.0 | -14.93% | 8998 | +27.18% |
| [+ Chunked Cross Entropy（分块交叉熵）](https://pytorch.org/torchtune/main/generated/torchtune.modules.loss.CEWithChunkedOutputLoss.html) | 42.9 | -15.83% | 9174 | +1.96% |
| [+ Activation Checkpointing（激活重计算）](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html#activation-checkpointing) | 24.9 | -41.93% | 7210 | -21.41% |
| [+ 优化器单步融合至反向传播](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html#fusing-optimizer-step-into-backward-pass) | 23.1 | -7.29% | 7309 | +1.38% |
| [+ Activation Offloading（激活值卸载）](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html#activation-offloading) | 21.8 | -5.48% | 7301 | -0.11% |
| [+ 8-bit AdamW 优化器](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html#lower-precision-optimizers) | 17.6 | -19.63% | 6960 | -4.67% |
| [LoRA](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html#glossary-lora) | 8.5 | -51.61% | 8210 | +17.96% |
| [QLoRA](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html#quantized-low-rank-adaptation-qlora) | 4.6 | -45.71% | 8035 | -2.13% |

相比 Baseline + Packed Dataset，最终行配置**显存降低了 81.9%**，且吞吐量提升了 **284.3%**。

<details>
<summary>点击展开复现最终行配置的命令</summary>

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

## 安装指南 🛠️

torchtune 经过针对 PyTorch 稳定版及 Nightly 预览版的严格测试，并深度整合了 `torchvision`（用于多模态 VLM）与 `torchao`（用于前沿量化技术）：

### 安装稳定版

```bash
# 安装 PyTorch, torchvision, torchao 稳定版
pip install torch torchvision torchao
pip install torchtune
```

### 安装 Nightly 预览版

```bash
# 安装 PyTorch, torchvision, torchao nightly 版本
pip install --pre --upgrade torch torchvision torchao --index-url https://download.pytorch.org/whl/nightly/cu126 # 可选项包含 cpu/cu118/cu124/cu126/xpu/rocm6.2/rocm6.3/rocm6.4
pip install --pre --upgrade torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

更多安装选项（包括从源码构建）请参见[安装文档](https://pytorch.org/torchtune/main/install.html)。

&nbsp;

安装完成后，运行如下命令校验安装：

```bash
tune --help
```

控制台将输出帮助信息：

```bash
usage: tune [-h] {ls,cp,download,run,validate} ...

Welcome to the torchtune CLI!

options:
  -h, --help            show this help message and exit

...
```

&nbsp;

## 快速上手 🚀

初次使用 torchtune，建议阅读[首个微调实战教程](https://pytorch.org/torchtune/main/tutorials/first_finetune_tutorial.html)以及涵盖评估、量化与推理的[端到端工作流教程](https://pytorch.org/torchtune/main/tutorials/e2e_flow.html)。以下以 Llama 3.1 为例进行演示。

### 1. 下载模型权重

访问 [`meta-llama`](https://huggingface.co/meta-llama) 官方 Hugging Face 页面获取权重授权后，使用如下命令一键下载模型权重、分词器及安全指引：

```bash
tune download meta-llama/Meta-Llama-3.1-8B-Instruct \
--output-dir /tmp/Meta-Llama-3.1-8B-Instruct \
--ignore-patterns "original/consolidated.00.pth" \
--hf-token <HF_TOKEN>
```

> [!Tip]
> 设置环境变量 `HF_TOKEN` 或在命令行中传入 `--hf-token` 进行鉴权认证。Token 可在 https://huggingface.co/settings/tokens 获取。

### 2. 运行微调 Recipe

在单张 GPU 上使用 LoRA 微调 Llama 3.1 8B：

```bash
tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device
```

若需执行分布式多卡训练，tune CLI 原生集成了 [torchrun](https://pytorch.org/docs/stable/elastic/run.html)。例如在 2 张 GPU 上全量微调 Llama 3.1 8B：

```bash
tune run --nproc_per_node 2 full_finetune_distributed --config llama3_1/8B_full
```

> [!Tip]
> 请务必将所有 torchrun 参数放置在 Recipe 名称**之前**；放置在之后的参数将仅用于覆盖 YAML 配置项，不会传递给底层分布式启动器。

### 3. 修改训练配置

修改配置支持两种直观方式：

**方式 A：命令行即时覆盖 (Config Overrides)**

直接在命令行中覆盖配置字段：

```bash
tune run lora_finetune_single_device \
--config llama2/7B_lora_single_device \
batch_size=8 \
enable_activation_checkpointing=True \
max_steps_per_epoch=128
```

**方式 B：复制到本地文件定制**

将官方预置配置复制到本地工作目录：

```bash
tune cp llama3_1/8B_full ./my_custom_config.yaml
Copied to ./my_custom_config.yaml
```

随后直接执行本地配置文件：

```bash
tune run full_finetune_distributed --config ./my_custom_config.yaml
```

运行 `tune --help` 可查看全部 CLI 命令细节，更多进阶用法参见[配置深度解析](https://pytorch.org/torchtune/main/deep_dives/configs.html)。

### 自定义数据集

torchtune 支持多样化的数据集格式微调，涵盖[指令微调数据集 (Instruct)](https://pytorch.org/torchtune/main/basics/instruct_datasets.html)、[多轮对话数据集 (Chat)](https://pytorch.org/torchtune/main/basics/chat_datasets.html) 以及[偏好对齐数据集 (Preference)](https://pytorch.org/torchtune/main/basics/preference_datasets.html) 等。详见[数据集 API 文档](https://pytorch.org/torchtune/main/api_ref_datasets.html)。

### 自定义计算设备支持

torchtune 支持在多种异构计算设备上运行，包括 NVIDIA GPU、Intel XPU、AMD ROCm、Apple MPS 以及华为昇腾 Ascend NPU。例如在 Intel XPU 上运行：

步骤 1：按照 [Intel GPU 入门指南](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) 配置环境。

步骤 2：通过命令行直接指定 `device=xpu`：
```bash
tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device device=xpu
```
或在本地配置文件中将 `device: cuda` 替换为 `device: xpu`。

&nbsp;

## 生态社区 🌍

torchtune 深度融入 PyTorch 与开源生态，紧密集成了众多知名工具：

- [Hugging Face Hub](https://huggingface.co/docs/hub/en/index)：用于[模型权重下载与访问](torchtune/_cli/download.py)
- [EleutherAI LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness)：用于微调模型的[基准评测](recipes/eleuther_eval.py)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/en/index)：用于[加载与处理训练/测试数据集](torchtune/datasets/_instruct.py)
- [PyTorch FSDP2](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)：用于大规模分布式训练
- [torchao](https://github.com/pytorch/ao)：用于低精度算子及[训练后量化 (Post-training Quantization)](recipes/quantize.py)
- [Weights & Biases](https://wandb.ai/site)：用于[日志追踪](https://pytorch.org/torchtune/main/deep_dives/wandb_logging.html)与指标看板
- [Comet](https://www.comet.com/site/)：用于训练监控与实验记录
- [ExecuTorch](https://pytorch.org/executorch-overview)：用于端侧与边缘设备[轻量化部署推理](https://github.com/pytorch/executorch/tree/main/examples/models/llama2#optional-finetuning)
- [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index)：用于低显存优化器
- [PEFT](https://github.com/huggingface/peft)：用于在 Hugging Face 生态中继续微调或部署推理

&nbsp;

### 社区贡献致谢

我们非常珍视社区的贡献，特别感谢以下开发者：
- [@SalmanMohammadi](https://github.com/salmanmohammadi)：贡献了基于 PPO 的 [RLHF 端到端微调 Recipe](recipes/ppo_full_finetune_single_device.py)
- [@fyabc](https://github.com/fyabc)：贡献了 Qwen2 模型、分词器及对应微调集成
- [@solitude-alive](https://github.com/solitude-alive)：贡献了 [Gemma 2B 模型](torchtune/models/gemma/)、Recipe 适配及数值精度校验
- [@yechenzhi](https://github.com/yechenzhi)：贡献了 [直接偏好优化 (DPO)](recipes/lora_dpo_single_device.py) 的完整实现与配置
- [@Optimox](https://github.com/Optimox)：贡献了全部 [Gemma2 系列变体模型](torchtune/models/gemma2)！

&nbsp;

## 致谢 🙏

本项目中的 Transformer 实现灵感来源于 [Llama2 官方源码](https://github.com/meta-llama/llama/blob/main/llama/model.py)。由衷感谢 EleutherAI、Hugging Face 与 Weights & Biases 团队的卓越合作。同时致敬开源生态中诸多优秀的库与工具：
- [gpt-fast](https://github.com/meta-pytorch/gpt-fast)：为我们开箱即用的高性能 LLM 推理提供了基石
- [llama recipes](https://github.com/meta-llama/llama-recipes)：推动了 Llama2 开源社区的繁荣
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)：为 PyTorch 生态带来了出色的低显存技术
- [@winglian](https://github.com/winglian/) 与 [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)：在早期设计与功能演进中提供了宝贵的反馈与讨论
- [lit-gpt](https://github.com/Lightning-AI/litgpt)：共同推进大模型微调生态的发展
- [HF TRL](https://github.com/huggingface/trl)：降低了奖励建模在 PyTorch 社区的使用门槛

&nbsp;

## 项目引用 📝

如果您在科研或项目中使用了 torchtune，请按如下格式引用：

```bibtex
@software{torchtune,
  title = {torchtune: PyTorch's finetuning library},
  author = {torchtune maintainers and contributors},
  url = {https://github.com/pytorch/torchtune},
  license = {BSD-3-Clause},
  month = apr,
  year = {2024}
}
```

&nbsp;

## 开源许可

torchtune 基于 [BSD-3-Clause 许可证](./LICENSE) 开源。请注意，使用第三方模型（如 Meta Llama 等）仍需遵守对应服务条款及许可协议。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月13日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
