.. _qat_distributed_recipe_label:

=============================================
Distributed Quantization-Aware Training (QAT)
=============================================

QAT allows for taking advantage of memory-saving optimisations from quantisation at inference time, without significantly
degrading model performace. In torchtune, we use `torchao <https://github.com/pytorch/ao>`_ to implement QAT during training.
This works by :ref:`simluating quantization numerics during fine-tuning <what_is_qat_label>`. While this may introduce memory and
compute overheads during training, our tests found that QAT significantly reduced performance degredation in evaluations of
quantized model, without compromising on model size reduction gains.

Interested in using this recipe? Check out some of our tutorials which show how it is used:

* :ref:`qat_finetune_label`

The best way to get started with our recipes is through the :ref:`cli_label`, which allows you to start fine-tuning
one of our built-in models without touching a single line of code!

For example, if you're interested in using this recipe with the latest `Llama models <https://llama.meta.com/>`_, you can fine-tune
in just two steps:

.. note::

    You may need to be granted access to the LLama model you're interested in. See
    :ref:`here <download_llama_label>` for details on accessing gated repositories.


.. code-block:: bash

    tune download meta-llama/Meta-Llama-3-8B-Instruct  \
    --output-dir /tmp/Meta-Llama-3-8B-Instruct \
    --ignore-patterns "original/consolidated.00.pth" \
    --HF_TOKEN <HF_TOKEN>

    tune run --nproc_per_node 4 qat_distributed \
    --config llama3/8B_qat_full

.. note::
  This workload requires at least 6 GPUs, each with VRAM of at least 80GB.

.. note::

    The :ref:`cli_label` allows you to list all our recipes and configs, run recipes, copy configs and recipes,
    and validate configs without touching a line of code!


Most of you will want to twist, pull, and bop all the different levers and knobs we expose in our recipes. Check out our
:ref:`configs tutorial <config_tutorial_label>` to learn how to customize recipes to suit your needs.

Currently, the main lever you can pull for QAT is by using delayed fake quantization:

Delayed fake quantization allows for control over the step after which fake quantization occurs.
Empirically, allowing the model to finetune without fake quantization initially allows the
weight and activation values to stabilize before fake quantizing them, potentially leading
to improved quantized accuracy. This can be specified through ``fake_quant_after_n_steps``. To
provide you with an idea of how to roughly configure this parameter, we've tested with
``fake_quant_after_n_steps ~= total_steps // 2``.

In the future we plan to support different quantization strategies. For now, note that you'll need at least
``torch>=2.4.0`` to use the `Int8DynActInt4WeightQATQuantizer <https://github.com/pytorch/ao/blob/08024c686fdd3f3dc2817094f817f54be7d3c4ac/torchao/quantization/prototype/qat/api.py#L35>`_
strategy. Generally, we apply QAT in three steps:

#. Run the ``qat_distributed`` recipe using the above command, or by following the tutorial. By default, this will use ``Int8DynActInt4WeightQATQuantizer``.
#. This produces an unquantized model in the original data type. To get an actual quantized model, follow this with
   ``tune run quantize`` while specifying the same quantizer in the config, e.g.

   .. code-block:: yaml

     # QAT specific args
     quantizer:
       _component_: torchtune.utils.quantization.Int8DynActInt4WeightQATQuantizer
       groupsize: 256

#. :ref:`Evaluate<qat_eval_label>` or `run inference <https://github.com/pytorch/torchtune/blob/main/recipes/quantization.md#generate>`_
   using your your quantized model by specifying the corresponding post-training quantizer:

   .. code-block:: yaml

     quantizer:
       _component_: torchtune.utils.quantization.Int8DynActInt4WeightQuantizer
       groupsize: 256

As with all of our recipes, you can also:

* Adjust :ref:`model precision <glossary_precision>`.
* Use :ref:`activation checkpointing <glossary_act_ckpt>`.
* Enable :ref:`gradient accumulation <glossary_grad_accm>`.
* Use :ref:`lower precision optimizers <glossary_low_precision_opt>`.


If you're interested in an overview of our memory optimisation features, check out our  :ref:`memory optimization overview<memory_optimisation_overview_label>`!
