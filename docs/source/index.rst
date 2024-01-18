Welcome to the TorchTune Documentation
=======================================

**TorchTune** is a Native-PyTorch library for LLM fine-tuning.

Getting Started
~~~~~~~~~~~~~~~

Topics in this section will help you get started with TorchTune.

.. grid:: 2

     .. grid-item-card:: :octicon:`file-code;1em`
        What is TorchTune?
        :img-top: _static/img/card-background.svg
        :link: intro-overview.html
        :link-type: url

        A gentle introduction to TorchTune. In this section,
        you will learn about main features of TorchTune
        and how you can use them in your projects.

     .. grid-item-card:: :octicon:`file-code;1em`
        Getting started with TorchTune
        :img-top: _static/img/card-background.svg
        :link: getting-started-setup.html
        :link-type: url

        A step-by-step tutorial on how to get started with
        TorchTune.

Tutorials
~~~~~~~~~

Ready to experiment? Check out some of the interactive
TorchTune tutorials.

.. customcardstart::

.. customcarditem::
   :header: Template Tutorial
   :card_description: A template tutorial. To be deleted.
   :image: _static/img/generic-pytorch-logo.png
   :link: generated_examples/template_tutorial.html
   :tags: Template

.. customcardend::

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Introduction
   :hidden:

   intro-overview

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting-started-setup

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   generated_examples/template_tutorial
   generated_examples/plot_the_best_example_in_the_world

Model Architectures
-------------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    torchtune.models.llama2_7b

Modeling Components and Building Blocks
---------------------------------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    torchtune.modules.CausalSelfAttention
    torchtune.modules.FeedForward
    torchtune.modules.KVCache
    torchtune.modules.RotaryPositionalEmbeddings
    torchtune.modules.RMSNorm
    torchtune.modules.Tokenizer
    torchtune.modules.TransformerDecoderLayer
    torchtune.modules.TransformerDecoder


PEFT Components

.. autosummary::
    :toctree: generated/
    :template: class.rst

    torchtune.modules.peft.LoRALinear
