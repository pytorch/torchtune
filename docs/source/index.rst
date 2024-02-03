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
        :link: overview.html
        :link-type: url

        A gentle introduction to TorchTune. In this section,
        you will learn about main features of TorchTune
        and how you can use them in your projects.

     .. grid-item-card:: :octicon:`file-code;1em`
        Installation instructions
        :img-top: _static/img/card-background.svg
        :link: install.html
        :link-type: url

        A step-by-step tutorial on how to install TorchTune.

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


.. ----------------------------------------------------------------------
.. Below is the toctree i.e. it defines the content of the left sidebar.
.. Each of the entry below corresponds to a file.rst in docs/source/.
.. ----------------------------------------------------------------------

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   overview
   install


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Recipes
   :hidden:

   recipes/how_to_run
   recipes/finetune_llm
   recipes/alpaca_generate

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   generated_examples/template_tutorial

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api_ref_datasets
   api_ref_models
   api_ref_modules
   api_ref_utilities
