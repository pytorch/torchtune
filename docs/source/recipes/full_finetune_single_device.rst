.. _recipe_label_here:

=====================
TEMPLATE Recipe Title
=====================

<Recipe intro>

Interested in using this recipe? Check out some of our tutorials which show how it is used:

.. Don't have any tutorials to reference? Consider writing one! : )

.. these tutorials are probably generic enough to be referenced in most of our recipes
.. but please consider if this is the case when writing this document.

* :ref:`finetune_llama_label`
* :ref:`e2e_flow`

The best way to get started with our recipes is through the :ref:`cli_label`, which allows you to start fine-tuning
one of our built-in models without touching a single line of code!

For example, if you're interested in using this recipe with the latest `Llama models <https://llama.meta.com/>`_, you can fine-tune
in just two steps:

.. fill the commands below out if you so desire

.. note::

    You may need to be granted access to the LLama model you're interested in. See
    :ref:`here <download_llama_label>` for details on accessing gated repositories.


.. code-block:: bash

    tune download path/to/huggingface_model \
        --output-dir /tmp/model_name \
        --hf-token <HF ACCESS TOKEN>

    tune run <recipe> --config <config> \
        model._component_=<model_component> \
        tokenizer._component_=<model_component> \
        checkpointer ... \
        logger._component_=torchtune.utils.metric_logging.WandBLogger \
        dataset ... \


.. note::

    The :ref:`cli_label` allows you to list all our recipes and configs, run recipes, copy configs and recipes,
    and validate configs without touching a line of code!


.. detail the recipe params below. you might want to include these defaults:

.. you can include this line for all recipes

Most of you will want to twist, pull, and bop all the different levers and knobs we expose in our recipes. Check out our
:ref:`configs tutorial <config_tutorial_label>` to learn how to customize recipes to suit your needs.
Are you also interested in our memory optimisation features? Check out our  :ref:`memory optimization overview<memory_optimisation_overview_label>`!
.. and for lora/qlora recipes

This recipe in particular supports :ref:`parameter efficient fine-tuning (PEFT) <glossary_peft>`: :ref:`glossary_lora` and :ref:`glossary_qlora`.

.. and for single device recipes

As with all our single-device recipes, you can also:

* Adjust :ref:`model precision <glossary_precision>`.
* Use :ref:`activation checkpointing <glossary_act_ckpt>`.
* Enable :ref:`gradient accumulation <glossary_grad_accm>`.
* Use :ref:`lower precision optimizers <glossary_low_precision_opt>`.


.. and you can add the below for LoRA
.. However, note that since LoRA significantly reduces memory usage due to gradient state, you will likely not need this
.. feature.

.. and for distributed recipes

As with all our distributed recipes:

* `glossary_distrib`
