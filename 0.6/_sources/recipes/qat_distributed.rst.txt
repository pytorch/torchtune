.. _qat_distributed_recipe_label:

=============================================
Distributed Quantization-Aware Training (QAT)
=============================================

QAT allows for taking advantage of memory-saving optimizations from quantization at inference time, without significantly
degrading model performance. In torchtune, we use `torchao <https://github.com/pytorch/ao>`_ to implement QAT.
This works by :ref:`simulating quantization numerics during fine-tuning <what_is_qat_label>`. While this may introduce memory and
compute overheads during training, our tests found that QAT significantly reduced performance degradation in evaluations of
quantized model, without compromising on model size reduction gains. Please see the `PyTorch blogpost <https://pytorch.org/blog/quantization-aware-training/>`_
on QAT for a deeper dive on how the technique works.

We provide pre-tested out-of-the-box configs which you can get up and running with the latest `Llama models <https://llama.meta.com/>`_
in just two steps:

.. code-block:: bash

    tune download meta-llama/Meta-Llama-3-8B-Instruct  \
    --output-dir /tmp/Meta-Llama-3-8B-Instruct \
    --ignore-patterns "original/consolidated.00.pth" \
    --HF_TOKEN <HF_TOKEN>

    tune run --nproc_per_node 6 qat_distributed \
    --config llama3/8B_qat_full

.. note::
  You may need to be granted access to the Llama model you're interested in. See
  :ref:`here <download_llama_label>` for details on accessing gated repositories.
  Also, this workload requires at least 6 GPUs, each with VRAM of at least 80GB e.g. A100s or H100s.

Currently, the main lever you can pull for QAT is by using *delayed fake quantization*.
Delayed fake quantization allows for control over the step after which fake quantization occurs.
Empirically, allowing the model to finetune without fake quantization initially allows the
weight and activation values to stabilize before fake quantizing them, potentially leading
to improved quantized accuracy. This can be specified through ``fake_quant_after_n_steps``. To
provide you with an idea of how to roughly configure this parameter, we've achieved best results with
``fake_quant_after_n_steps ~= total_steps // 2``.

In the future we plan to support different quantization strategies. For now, note that you'll need at least
``torch>=2.4.0`` to use the `Int8DynActInt4WeightQATQuantizer <https://github.com/pytorch/ao/blob/08024c686fdd3f3dc2817094f817f54be7d3c4ac/torchao/quantization/prototype/qat/api.py#L35>`_
strategy. Generally, the pipeline for training, quantizing, and evaluating a model using QAT is:

#. Run the ``qat_distributed`` recipe using the above command, or by following the tutorial. By default, this will use ``Int8DynActInt4WeightQATQuantizer``.
#. This produces an un-quantized model in the original data type. To get an actual quantized model, follow this with
   ``tune run quantize`` while specifying the same quantizer in the config, e.g.

   .. code-block:: yaml

     # QAT specific args
     quantizer:
       _component_: torchtune.training.quantization.Int8DynActInt4WeightQATQuantizer
       groupsize: 256

#. :ref:`Evaluate<qat_eval_label>` or `run inference <https://github.com/pytorch/torchtune/blob/main/recipes/quantization.md#generate>`_
   using your your quantized model by specifying the corresponding post-training quantizer:

   .. code-block:: yaml

     quantizer:
       _component_: torchtune.training.quantization.Int8DynActInt4WeightQuantizer
       groupsize: 256

.. note::

  We're using config files to show how to customize the recipe in these examples. Check out the
  :ref:`configs tutorial <config_tutorial_label>` to learn more.

Many of our other memory optimization features can be used in this recipe, too:

* Adjust :ref:`model precision <glossary_precision>`.
* Use :ref:`activation checkpointing <glossary_act_ckpt>`.
* Enable :ref:`gradient accumulation <glossary_grad_accm>`.
* Use :ref:`lower precision optimizers <glossary_low_precision_opt>`.

You can learn more about all of our memory optimization features in our  :ref:`memory optimization overview<memory_optimization_overview_label>`.

Interested in seeing this recipe in action? Check out some of our tutorials to show off how it can be used:

* :ref:`qat_finetune_label`
