.. _overview_label:

==================
TorchTune Overview
==================

On this page, we'll walk through an overview of TorchTune, including features, key concepts and additional pointers.

What is TorchTune?
------------------

TorchTune is a PyTorch library for easily authoring, fine-tuning and experimenting with LLMs. The library emphasizes 4 key aspects:

- **Simplicity and Extensibility**. Native-PyTorch, componentized design and easy-to-reuse abstractions
- **Correctness**. High-bar on proving the correctness of components and recipes
- **Stability**. PyTorch just works. So should TorchTune
- **Democratizing LLM fine-tuning**. Works out-of-the-box on different hardware


TorchTune provides:

- Modular native-PyTorch implementations of popular LLMs
- Interoperability with popular model zoos through checkpoint-conversion utilities
- Training recipes for a variety of fine-tuning techniques
- Integration with `HuggingFace Datasets <https://huggingface.co/docs/datasets/en/index>`_ for training and `EleutherAI's Eval <https://github.com/EleutherAI/lm-evaluation-harness>`_ Harness for evaluation
- Support for distributed training using `FSDP <https://pytorch.org/docs/stable/fsdp.html>`_
- Yaml configs for easily configuring training runs

Excited? To get started, checkout some of our tutorials, including:

- our :ref:`full finetuning tutorial <finetune_llama_label>` to get started and finetune your first LLM using TorchTune.
- our :ref:`LoRA tutorial <lora_finetune_label>` to learn about parameter-efficient finetuning with TorchTune.

Key Concepts
------------

As you go through the tutorials and code, there are two concepts which will help you better understand and use TorchTune.

**Configs.** Yaml files which help you configure training settings (dataset, model, chekckpoint) and
hyperparameters (batch size, learning rate) without modifying code.
See the "Getting Started with Configs" tutorial for more information.

**Recipes.** Recipes can be thought of
as "targeted" end-to-end pipelines for training and optionally evaluating LLMs.
Each recipe implements a training method (eg: full fine-tuning) with a set of meaningful
features (eg: FSDP + Activation Checkpointing + Gradient Accumulation + Mixed Precision training)
applied to a given model family (eg: Llama2). See the tutorial on :ref:`Training Recipe Deep-Dive<recipe_deepdive>` for more information.

|

Design Principles
-----------------

TorchTune embodies `PyTorch’s design philosophy <https://pytorch.org/docs/stable/community/design.html>`_, especially "usability over everything else".

**Native PyTorch**

TorchTune is a native-PyTorch library. While we provide integrations with the surrounding ecosystem (eg: HuggingFace Datasets, EluetherAI Eval Harness), all of the core functionality is written in PyTorch.


**Simplicity and Extensibility**

TorchTune is designed to be easy to understand, use and extend.

- Composition over implementation inheritance - layers of inheritance for code re-use makes the code hard to read and extend
- No training frameworks - explicitly outlining the training logic makes it easy to extend for custom use cases
- Code duplication is prefered over unecessary abstractions
- Modular building blocks over monolithic components


**Correctness**

TorchTune provides well-tested components with a high-bar on correctness. The library will never be the first to provide a feature, but available features will be thoroughly tested. We provide

- Extensive unit-tests to ensure component-level numerical parity with reference implementations
- Checkpoint-tests to ensure model-level numerical parity with reference implementations
- Integration tests to ensure recipe-level performance parity with reference implementations on standard benchmarks
