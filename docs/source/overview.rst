==================
TorchTune Overview
==================

What is TorchTune?
------------------

TorchTune is a native-Pytorch library for easily authoring, fine-tuning and experimenting with LLMs.

The library provides:

- Native-PyTorch implementations of popular LLMs, with convertors to transform checkpoints into TorchTune's format
- Training recipes for popular fine-tuning techniques with reference benchmarks and comprehensive correctness checks
- Integration with HuggingFace Datasets for training and EleutherAI's Eval Harness for evaluation
- Support for distributed training using FSDP from PyTorch Distributed
- Yaml configs for easily configuring training runs


Design Principles
-----------------

TorchTune embodies `PyTorch’s design philosophy <https://pytorch.org/docs/stable/community/design.html>`_, especially "usability over everything else".

|

**Native PyTorch**

TorchTune is a native-PyTorch library. While we provide integrations with the surrounding ecosystem (eg: HuggingFace Datasets, EluetherAI Eval Harness), all of the core functionality is written in PyTorch.

|

**Simplicity and Extensibility**

TorchTune is designed to be easy to understand, use and extend.

- Composition over implementation inheritance - layers of inheritance for code re-use makes the code hard to read and extend
- No training frameworks - explicitly outlining the training logic makes it easy to extend for custom use cases
- Code duplication is prefered over unecessary abstractions
- Modular building blocks over monolithic components

|

**Correctness**

TorchTune provides well-tested components with a high-bar on correctness. The library will never be the first to provide a feature, but available features will be thoroughly tested. We provide

- Extensive unit-tests to ensure component-level numerical parity with reference implementations
- Checkpoint-tests to ensure model-level numerical parity with reference implementations
- Integration tests to ensure recipe-level performance parity with reference implementations on standard benchmarks
