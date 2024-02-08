Welcome to the TorchTune Documentation
=======================================

**TorchTune** is a Native-PyTorch library for LLM fine-tuning.

Getting Started
~~~~~~~~~~~~~~~

Topics in this section will help you get started with TorchTune.

.. grid:: 3

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

     .. grid-item-card:: :octicon:`file-code;1em`
         Finetune your first model
         :img-top: _static/img/card-background.svg
         :link: examples/first_finetune_tutorial
         :link-type: url

         Follow a simple tutorial to finetune Llama2 with TorchTune.

Tutorials
~~~~~~~~~

Ready to experiment? Check out some of the interactive
TorchTune tutorials.

.. customcardstart::

.. customcarditem::
   :header: Finetune Llama2 with TorchTune
   :card_description: Quickstart to get you finetuning an LLM fast.
   :image: _static/img/generic-pytorch-logo.png
   :link: examples/first_finetune_tutorial.html
   :tags: finetuning,llama2

.. customcarditem::
   :header: Training Recipes Deep-Dive
   :card_description: Dive into TorchTune's Training Recipes
   :image: _static/img/generic-pytorch-logo.png
   :link: examples/recipe_deepdive.html
   :tags: finetuning

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

   examples/recipe_deepdive
   examples/first_finetune_tutorial

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api_ref_datasets
   api_ref_models
   api_ref_modules
   api_ref_utilities
