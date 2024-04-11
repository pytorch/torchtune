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

        A gentle introduction to TorchTune and how you can
        use the library in your projects.

     .. grid-item-card:: :octicon:`file-code;1em`
        Installation instructions
        :img-top: _static/img/card-background.svg
        :link: install.html
        :link-type: url

        A step-by-step tutorial on how to install TorchTune.

     .. grid-item-card:: :octicon:`file-code;1em`
         Finetune your first model
         :img-top: _static/img/card-background.svg
         :link: examples/first_finetune_tutorial.html
         :link-type: url

         Follow a simple tutorial to finetune Llama2 with TorchTune.

Tutorials
~~~~~~~~~

Ready to experiment? Check out some of the interactive
TorchTune tutorials.

.. customcardstart::

.. customcarditem::
   :header: LLM Full Finetuning Recipe
   :card_description: Full Finetuning for Llama2
   :image: _static/img/generic-pytorch-logo.png
   :link: examples/finetune_llm.html
   :tags: finetuning,llama2

.. customcarditem::
   :header: Finetuning with LoRA in TorchTune
   :card_description: Parameter-efficient finetuning of Llama2 using LoRA
   :image: _static/img/generic-pytorch-logo.png
   :link: examples/lora_finetune.html
   :tags: finetuning,llama2,lora

.. customcarditem::
   :header: Training Recipes Deep-Dive
   :card_description: Dive into TorchTune's Training Recipes
   :image: _static/img/generic-pytorch-logo.png
   :link: examples/recipe_deepdive.html
   :tags: finetuning

.. customcarditem::
   :header: Understanding the Checkpointer
   :card_description: Dive into TorchTune's Checkpointers
   :image: _static/img/generic-pytorch-logo.png
   :link: examples/checkpointer.html
   :tags: checkpointing

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
   examples/first_finetune_tutorial

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   examples/finetune_llm
   examples/lora_finetune
   examples/configs

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Deep-Dives
   :hidden:

   examples/recipe_deepdive
   examples/checkpointer

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api_ref_config
   api_ref_datasets
   api_ref_models
   api_ref_modules
   api_ref_utilities
