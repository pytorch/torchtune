.. _<recipe_name>_recipe_label:

============
Recipe Title
============

<Recipe intro>

Interested in using this recipe? Check out some of our tutorials which show how it is used:

.. Don't have any tutorials to reference? Consider writing one! : )

.. these tutorials are probably generic enough to be referenced in most of our recipes
.. but please consider if this is the case when writing this document.

* :ref:`finetune_llama_label`
* :ref:`e2e_flow`

The best way to get started with our recipes is through the :ref:`cli_label`, which allows you to
list all our recipes and configs, run recipes, copy configs and recipes, and validate configs
without touching a line of code!

For example, if you're interested in using this recipe with the latest `Llama models <https://llama.meta.com/>`_, you can fine-tune
in just two steps:

.. fill the commands below out if you so desire

.. note::

    You may need to be granted access to the Llama model you're interested in. See
    :ref:`here <download_llama_label>` for details on accessing gated repositories.


.. code-block:: bash

    tune download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output-dir /tmp/Meta-Llama-3.1-8B-Instruct \
    --ignore-patterns "original/consolidated.00.pth" \
    --HF_TOKEN <HF_TOKEN>

    tune run <recipe> --config <config> \
        model._component_=<model_component> \
        tokenizer._component_=<model_component> \
        checkpointer ... \
        logger._component_=torchtune.utils.metric_logging.WandBLogger \
        dataset ... \

.. detail the recipe params below. you might want to include these defaults:

.. you can include this line for all recipes

Most of you will want to twist, pull, and bop all the different levers, buttons, and knobs we expose in our recipes. Check out our
:ref:`configs tutorial <config_tutorial_label>` to learn how to customize recipes to suit your needs.

There are <> levers to pull when working with <>, specifically:

.. and for lora/qlora recipes
:ref:`Parameter efficient fine-tuning (PEFT) <glossary_peft>` using:

* :ref:`glossary_lora`
* :ref:`glossary_qlora`.

.. and generally for all our recipes:

As with all of our recipes, you can also:

* Adjust :ref:`model precision <glossary_precision>`.
* Use :ref:`activation checkpointing <glossary_act_ckpt>`.
* Enable :ref:`gradient accumulation <glossary_grad_accm>`.
* Use :ref:`lower precision optimizers <glossary_low_precision_opt>`.


.. and you can add the below for LoRA
.. However, note that since LoRA significantly reduces memory usage due to gradient state, you will likely not need this
.. feature.

.. and for distributed recipes

.. As with all our distributed recipes:

.. * `glossary_distrib`


If you're interested in an overview of our memory optimisation features, check out our  :ref:`memory optimization overview<memory_optimisation_overview_label>`!
