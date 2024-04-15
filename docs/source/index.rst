Welcome to the torchtune Documentation
=======================================

**torchtune** is a Native-PyTorch library for LLM fine-tuning.

Getting Started
~~~~~~~~~~~~~~~

Topics in this section will help you get started with torchtune.

.. grid:: 4

     .. grid-item-card:: :octicon:`file-code;1em`
        What is torchtune?
        :img-top: _static/img/card-background.svg
        :link: overview.html
        :link-type: url

        A gentle introduction to torchtune and how you can
        use the library in your projects.

     .. grid-item-card:: :octicon:`file-code;1em`
        Installation instructions
        :img-top: _static/img/card-background.svg
        :link: install.html
        :link-type: url

        A step-by-step tutorial on how to install torchtune.

     .. grid-item-card:: :octicon:`file-code;1em`
         Finetune your first model
         :img-top: _static/img/card-background.svg
         :link: tutorials/first_finetune_tutorial.html
         :link-type: url

         Follow a simple tutorial to finetune Llama2 with torchtune.

     .. grid-item-card:: :octicon:`file-code;1em`
         End-to-End Workflow with torchtune
         :img-top: _static/img/card-background.svg
         :link: tutorials/e2e_flow.html
         :link-type: url

         Train, Evaluate, Quantize and then Generate with your LLM.

Tutorials
~~~~~~~~~

Ready to experiment? Check out some of the interactive
torchtune tutorials.

.. customcardstart::

.. customcarditem::
   :header: Finetuning with LoRA in torchtune
   :card_description: Parameter-efficient finetuning of Llama2 using LoRA
   :image: _static/img/generic-pytorch-logo.png
   :link: tutorials/lora_finetune.html
   :tags: finetuning,llama2,lora

.. customcarditem::
   :header: End-to-End Workflow with torchtune
   :card_description: Train, Evaluate, Quantize and then Generate with your LLM.
   :image: _static/img/generic-pytorch-logo.png
   :link: tutorials/e2e_flow.html
   :tags: finetuning,quantization,inference,evals,llama2

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
   tutorials/first_finetune_tutorial
   tutorials/e2e_flow

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/lora_finetune
   tutorials/first_finetune_tutorial
   tutorials/e2e_flow

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Deep-Dives
   :hidden:

   deep_dives/checkpointer
   deep_dives/configs
   deep_dives/recipe_deepdive
   deep_dives/wandb_logging

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
