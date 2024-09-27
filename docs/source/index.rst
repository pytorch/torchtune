Welcome to the torchtune Documentation
=======================================

**torchtune** is a Native-PyTorch library for LLM fine-tuning.

Getting Started
~~~~~~~~~~~~~~~

Topics in this section will help you get started with torchtune.

.. grid:: 3

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

Tutorials
~~~~~~~~~

Ready to experiment? Check out some of the interactive
torchtune tutorials.

.. customcardstart::

.. customcarditem::
   :header: Llama3 in torchtune
   :card_description:
   :image: _static/img/generic-pytorch-logo.png
   :link: tutorials/llama3.html
   :tags: finetuning,llama3

.. customcarditem::
   :header: Finetuning with LoRA in torchtune
   :card_description: Parameter-efficient finetuning of Llama2 using LoRA
   :image: _static/img/generic-pytorch-logo.png
   :link: tutorials/lora_finetune.html
   :tags: finetuning,llama2,lora

.. customcarditem::
   :header: Understanding QLoRA in torchtune
   :card_description: Using QLoRA to quantize base model weights and maximize memory savings
   :image: _static/img/generic-pytorch-logo.png
   :link: tutorials/qlora_finetune.html
   :tags: finetuning,llama2,qlora

.. customcarditem::
   :header: Finetuning with QAT in torchtune
   :card_description: Finetuning of Llama3 using QAT
   :image: _static/img/generic-pytorch-logo.png
   :link: tutorials/qat_finetune.html
   :tags: finetuning,llama3,qat,quantization,evals

.. customcarditem::
   :header: End-to-End Workflow with torchtune
   :card_description: Train, Evaluate, Quantize and then Generate with your LLM.
   :image: _static/img/generic-pytorch-logo.png
   :link: tutorials/e2e_flow.html
   :tags: finetuning,quantization,inference,evals,llama2

.. customcarditem::
   :header: Distilling Llama3 8B into 1B
   :card_description: Distilling Llama3 8B into 1B using Knowledge Distillation
   :image: _static/img/generic-pytorch-logo.png
   :link: tutorials/llama_kd_tutorial.html
   :tags: finetuning,llama3,kd

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
   tune_cli

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Finetuning Recipes
   :hidden:

   recipes/recipes_overview
   recipes/lora_finetune_single_device
   recipes/qat_distributed

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Basics
   :hidden:

   basics/datasets_overview
   basics/chat_datasets
   basics/instruct_datasets
   basics/multimodal_datasets
   basics/preference_datasets
   basics/text_completion_datasets
   basics/model_transforms
   basics/messages
   basics/message_transforms
   basics/tokenizers
   basics/prompt_templates

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/llama3
   tutorials/chat
   tutorials/lora_finetune
   tutorials/qlora_finetune
   tutorials/qat_finetune
   tutorials/e2e_flow
   tutorials/datasets
   tutorials/memory_optimizations
   tutorials/llama_kd_tutorial

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Deep-Dives
   :hidden:

   deep_dives/checkpointer
   deep_dives/configs
   deep_dives/recipe_deepdive
   deep_dives/comet_logging
   deep_dives/wandb_logging

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api_ref_config
   api_ref_data
   api_ref_datasets
   api_ref_generation
   api_ref_models
   api_ref_modules
   api_ref_rlhf
   api_ref_training
   api_ref_utilities
