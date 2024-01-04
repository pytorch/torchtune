TorchTune
=========

This library is part of the `PyTorch
<http://pytorch.org/>`_ project. PyTorch is an open source
machine learning framework.

The :mod:`torchtune` package contains tools and utilities to finetune language models
with native PyTorch techniques.

Model Architectures

.. autosummary::
    :toctree: generated/
    :template: class.rst

    torchtune.models.llama2.models.llama2_7b


Modeling Components and Building Blocks

.. autosummary::
    :toctree: generated/
    :template: class.rst

    torchtune.models.llama2.transformer.TransformerDecoder

.. autosummary::
    :toctree: generated/
    :template: class.rst

    torchtune.models.llama2.transformer.TransformerDecoderLayer

.. autosummary::
    :toctree: generated/
    :template: class.rst

    torchtune.models.llama2.attention.LlamaSelfAttention

Tokenizers

.. autosummary::
    :toctree: generated/
    :template: class.rst

    torchtune.models.llama2.models.llama2_tokenizer


.. toctree::
    :maxdepth: 1
    :caption: Examples and tutorials

    auto_examples/index
