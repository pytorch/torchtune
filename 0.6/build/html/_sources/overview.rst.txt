.. _overview_label:

==================
torchtune Overview
==================

On this page, we'll walk through an overview of torchtune, including features, key concepts and additional pointers.

What is torchtune?
------------------

torchtune is a PyTorch library for easily authoring, fine-tuning and experimenting with LLMs. The library emphasizes 4 key aspects:

- **Simplicity and Extensibility**. Native-PyTorch, componentized design and easy-to-reuse abstractions
- **Correctness**. High bar on proving the correctness of components and recipes
- **Stability**. PyTorch just works. So should torchtune
- **Democratizing LLM fine-tuning**. Works out-of-the-box on different hardware


torchtune provides:

- Modular native-PyTorch implementations of popular LLMs
- Interoperability with popular model zoos through checkpoint-conversion utilities
- Training recipes for a variety of fine-tuning techniques
- Integration with `Hugging Face Datasets <https://huggingface.co/docs/datasets/en/index>`_ for training and `EleutherAI's Eval Harness <https://github.com/EleutherAI/lm-evaluation-harness>`_ for evaluation
- Support for distributed training using `FSDP2 <https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md>`_
- YAML configs for easily configuring training runs

Excited? To get started, checkout some of our tutorials, including:

- Our :ref:`quickstart guide <finetune_llama_label>` to finetune your first LLM using torchtune.
- Our :ref:`LoRA tutorial <lora_finetune_label>` to learn about parameter-efficient finetuning with torchtune.
- Our :ref:`QLoRA tutorial <qlora_finetune_label>` to attain maximal memory efficiency with torchtune.

You can check out our :ref:`recipes overview<recipes_overview_label>` to see all the fine-tuning techniques we support.

Key Concepts
------------

As you go through the tutorials and code, there are two concepts which will help you better understand and use torchtune.

**Configs.** YAML files which help you configure training settings (dataset, model, checkpoint) and
hyperparameters (batch size, learning rate) without modifying code.
See the ":ref:`All About Configs<config_tutorial_label>`" deep-dive for more information.

**Recipes.** Recipes can be thought of
as targeted end-to-end pipelines for training and optionally evaluating LLMs.
Each recipe implements a training method (eg: full fine-tuning) with a set of meaningful
features (eg: FSDP2 + Activation Checkpointing + Gradient Accumulation + Reduced Precision training)
applied to a given model family (eg: Llama3.1). See the ":ref:`What Are Recipes?<recipe_deepdive>`" deep-dive for more information.


.. _design_principles_label:

Design Principles
-----------------

torchtune embodies `PyTorch’s design philosophy <https://pytorch.org/docs/stable/community/design.html>`_, especially "usability over everything else".

**Native PyTorch**

torchtune is a native-PyTorch library. While we provide integrations with the surrounding ecosystem (eg: `Hugging Face Datasets <https://huggingface.co/docs/datasets/en/index>`_,
`EleutherAI's Eval Harness <https://github.com/EleutherAI/lm-evaluation-harness>`_), all of the core functionality is written in PyTorch.


**Simplicity and Extensibility**

torchtune is designed to be easy to understand, use and extend.

- Composition over implementation inheritance - layers of inheritance for code re-use makes the code hard to read and extend
- No training frameworks - explicitly outlining the training logic makes it easy to extend for custom use cases
- Code duplication is prefered over unecessary abstractions
- Modular building blocks over monolithic components


**Correctness**

torchtune provides well-tested components with a high bar on correctness. The library will never be the first to provide a feature, but available features will be thoroughly tested. We provide

- Extensive unit tests to ensure component-level numerical parity with reference implementations
- Checkpoint tests to ensure model-level numerical parity with reference implementations
- Integration tests to ensure recipe-level performance parity with reference implementations on standard benchmarks
