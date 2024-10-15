.. _memory_optimization_overview_label:

============================
Memory Optimization Overview
============================

**Author**: `Salman Mohammadi <https://github.com/SalmanMohammadi>`_

torchtune comes with a host of plug-and-play memory optimization components which give you lots of flexibility
to ``tune`` our recipes to your hardware. This page provides a brief glossary of these components and how you might use them.
To make things easy, we've summarized these components in the following table:

.. csv-table:: Memory optimization components
   :header: "Component", "When to use?"
   :widths: auto

   ":ref:`glossary_precision`", "You'll usually want to leave this as its default ``bfloat16``. If you're struggling with training stability or accuracy due to precision, fp32 may help, but will significantly increase memory usage and decrease training speed."
   ":ref:`glossary_act_ckpt`", "Use when you're memory constrained and need to handle larger batch sizes or longer context lengths. Be aware that it may slow down training speed."
   ":ref:`glossary_grad_accm`", "Helpful when memory-constrained to simulate larger batch sizes. Often preferable to activation checkpointing for better training speed."
   ":ref:`glossary_low_precision_opt`", "When you need to further reduce memory usage beyond using ``bf16`` by reducing the precision in the optimizer states. Note that lower precision optimizers may reduce training stability/accuracy."
   ":ref:`glossary_opt_in_bwd`", "Helps reduce memory usage when using stateful optimizers, particularly when full-finetuning large models with high gradient memory usage. This is not compatible with ``gradient_accumulation_steps``, so training may slow down due to reduced model throughput."
   ":ref:`glossary_lora`", "When you want to significantly reduce the number of trainable parameters, saving gradient and optimizer memory during training, and significantly speeding up training."
   ":ref:`glossary_qlora`", "When you need even more memory savings than LoRA, at the potential cost of some training speed. Useful for very large models or limited hardware."


.. note::

  In its current state, this tutorial is focused on single-device optimizations. Check in soon as we update this page
  for the latest memory optimization features for distributed fine-tuning.

.. _glossary_precision:


Model Precision
---------------

*What's going on here?*

We use the term "precision" to refer to the underlying data type used to represent the model and optimizer parameters.
We support two data types in torchtune:

.. note::

  We recommend diving into Sebastian Raschka's `blogpost on mixed-precision techniques <https://sebastianraschka.com/blog/2023/llm-mixed-precision-copy.html>`_
  for a deeper understanding of concepts around precision and data formats.

* ``fp32``, commonly referred to as "full-precision", uses 4 bytes per model and optimizer parameter.
* ``bfloat16``, referred to as "half-precision", uses 2 bytes per model and optimizer parameter - effectively half
  the memory of ``fp32``, and also improves training speed. Generally, if your hardware supports training with ``bfloat16``,
  we recommend using it - this is the default setting for our recipes.

.. note::

  Another common paradigm is "mixed-precision" training: where model weights are in ``bfloat16`` (or ``fp16``), and optimizer
  states are in ``fp32``. Currently, we don't support mixed-precision training in torchtune.

*Sounds great! How do I use it?*

Simply use the ``dtype`` flag or config entry in all our recipes! For example, to use half-precision training in ``bf16``,
set ``dtype=bf16``.

.. _glossary_act_ckpt:

Activation Checkpointing
------------------------

*What's going on here?*

The relevant section in the `PyTorch documentation <https://pytorch.org/docs/stable/checkpoint.html>`_ explains this concept well.
To quote:

  Activation checkpointing is a technique that trades compute for memory.
  Instead of keeping tensors needed for backward alive until they are used in
  gradient computation during backward, forward computation in checkpointed
  regions omits saving tensors for backward and recomputes them during the backward pass.

This setting is helpful for when you're memory-constrained, especially due to larger batch sizes or longer context lengths.
However, these savings in memory come at the cost of training speed (i.e. tokens-per-second),
and in most cases training can slow down quite a bit as a result of this activation recomputation.

*Sounds great! How do I use it?*

To enable activation checkpointing, use the ``enable_activation_checkpointing`` config entry or flag
in any of our recipes, e.g. ``enable_activation_checkpointing=True``.

.. _glossary_act_off:

Activation Offloading
---------------------

*What's going on here?*

You may have just read about activation checkpointing! Similar to checkpointing, offloading is a memory
efficiency technique that allows saving GPU VRAM by temporarily moving activations to CPU and bringing
them back when needed in the backward pass.

See `PyTorch autograd hook tutorial <https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html#saving-tensors-to-cpu>`_
for more details about how this is implemented through saved_tensors_hooks.

This setting is especially helpful for larger batch sizes, or longer context lengths when you're memory constrained.
While of course it takes runtime and resources to move Tensors from GPU to CPU and back, the implementation in
torchtune uses multiple CUDA streams (when available) in order to overlap the extra communication with the computation
to hide the extra runtime. As the communication workload is variable depending on the number and size of tensors being
offloaded, it is common to not offload every single activation. In fact, one can use offloading in conjunction with activations
checkpointing, where all activations will either be recomputed later in the backward or brought back from the CPU.

*Sounds great! How do I use it?*

To enable activation offloading, use the ``enable_activation_offloading`` config entry or flag
in our lora finetuning single device recipe, e.g. ``enable_activation_offloading=True``. To allow
usage of streams, make sure you are on a torch version later than PyTorch 2.5.0.dev20240907.

.. _glossary_grad_accm:

Gradient Accumulation
---------------------

*What's going on here?*

Gradient accumulation allows you to simulate large batch sizes by *accumulating* gradients over several
batches before updating model parameters using the optimizer. Concretely, the total number of samples used
for a gradient update is when using gradient accumulation is:

  ``total_batch_size = batch_size * gradient_accumulation_steps``

For example: with ``batch_size=1`` and ``gradient_accumulation_steps=32`` we get a total batch size of 32.

.. note::

  For other components in torchtune which use "steps", such as :ref:`metric logging <metric_logging_label>`, or
  :func:`learning rate schedulers <torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup>`, a "step" is counted as a
  single update to model parameters, rather than a single model forward pass with the data.
  Suppose ``gradient_accumulation_steps = 4`` and ``log_every_n_steps = 10``.
  Metrics would be logged every 10 global steps, which translates to every 40 model forward passes.
  For this reason, metric logging will appear less frequently when training with gradient accumulation,
  and progress bars may update more slowly.


If you're using one of our distributed recipes, simply multiply by the number of devices:

  ``total_batch_size = batch_size * gradient_accumulation_steps * num_devices``

Gradient accumulation is especially useful when you are memory constrained. In this case,
accumulating gradients might give you better training speed than enabling :ref:`activation
checkpointing <glossary_act_ckpt>`, since activation checkpointing reduces memory consumption at the cost of repeated
computations.

*Sounds great! How do I use it?*

All of our finetuning recipes support simulating larger batch sizes by accumulating gradients. Just set the
``gradient_accumulation_steps`` flag or config entry.

.. note::

  Gradient accumulation should always be set to 1 when :ref:`fusing the optimizer step into the backward pass <glossary_opt_in_bwd>`.

.. _glossary_low_precision_opt:

Lower Precision Optimizers
--------------------------

*What's going on here?*

In addition to :ref:`reducing model and optimizer precision <glossary_precision>` during training, we can further reduce precision in our optimizer states.
All of our single-device fine-tuning recipes support lower-precision optimizers from the `bitsandbytes <https://huggingface.co/docs/bitsandbytes/main/en/index>`_ library -
a good place to start might be the ``AdamW8bit`` and ``PagedAdamW8bit`` optimizers, which we've tested our recipes with.

*Sounds great! How do I use it?*

To use this in your recipes, make sure you have installed bitsandbytes (``pip install bitsandbytes``). Then, enable
a low precision optimizer using the :ref:`cli_label`:

.. code-block:: bash

  tune run <RECIPE> --config <CONFIG> \
  optimizer=bitsandbytes.optim.PagedAdamW

or by directly :ref:`modifying a config file<config_tutorial_label>`:

.. code-block:: yaml

  optimizer:
    _component_: bitsandbytes.optim.PagedAdamW
    lr: 2e-5

.. _glossary_opt_in_bwd:

Fusing Optimizer Step into Backward Pass
----------------------------------------

*What's going on here?*

Stateful optimizers (e.g. optimizers which use momentum) are the default in modern deep learning due to their stable convergence properties.
However, maintaining a state of gradient statistics comes at the cost of additional memory usage. An immediate alternative might be to
turn to stateless optimizers such as `stochastic gradient descent <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_
without momentum, which don't require any additional memory usage, but will likely result in worse convergence during training.

Can we find a middle ground here? Let's consider a technique which enables the use of "stateful" optimizers such as `AdamW <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>`_
without the memory overhead of gradient statistics, and without sacrificing their desirable convergence properties.
How is this possible, you might ask? By *completely removing the buffer of gradients* which are stored by the optimizer during its ``step()``.

To understand how this works, we encourage you to read through the relevant PyTorch tutorial on this concept:
`How to save memory by fusing the optimizer step into the backward pass <https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html>`_.


*Sounds great! How do I use it?*

.. todo ref full finetune recipe doc

In torchtune, you can enable this feature using the ``optimizer_in_bwd`` flag, which is currently only supported in our
single-device full finetune recipe. This feature works best when gradient memory is particularly large;
e.g. when using a stateful optimizer with a model with a lot of parameters, and when you don't need to use
:ref:`gradient accumulation <glossary_grad_accm>`.

.. _glossary_peft:

Parameter Efficient Fine-Tuning (PEFT)
--------------------------------------

.. _glossary_lora:

Low Rank Adaptation (LoRA)
^^^^^^^^^^^^^^^^^^^^^^^^^^


*What's going on here?*

You can read our tutorial on :ref:`finetuning Llama2 with LoRA<lora_finetune_label>` to understand how LoRA works, and how to use it.
Simply stated, LoRA greatly reduces the number of trainable parameters, thus saving significant gradient and optimizer
memory during training.

*Sounds great! How do I use it?*

You can finetune using any of our recipes with the ``lora_`` prefix, e.g. :ref:`lora_finetune_single_device<lora_finetune_recipe_label>`. These recipes utilize
LoRA-enabled model builders, which we support for all our models, and also use the ``lora_`` prefix, e.g.
the :func:`torchtune.models.llama3.llama3` model has a corresponding :func:`torchtune.models.llama3.lora_llama3`.
We aim to provide a comprehensive set of configurations to allow you to get started with training with LoRA quickly,
just specify any config with ``_lora`` in its name, e.g:

.. code-block:: bash

  tune run lora_finetune_single_device --config llama3/8B_lora_single_device


There are two sets of parameters to customize LoRA to suit your needs. Firstly, the parameters which control
which linear layers LoRA should be applied to in the model:

* ``lora_attn_modules: List[str]`` accepts a list of strings specifying which layers of the model to apply
  LoRA to:

  * ``q_proj`` applies LoRA to the query projection layer.
  * ``k_proj`` applies LoRA to the key projection layer.
  * ``v_proj`` applies LoRA to the value projection layer.
  * ``output_proj`` applies LoRA to the attention output projection layer.

  Whilst adding more layers to be fine-tuned may improve model accuracy,
  this will come at the cost of increased memory usage and reduced training speed.

* ``apply_lora_to_mlp: Bool`` applies LoRA to the MLP in each transformer layer.
* ``apply_lora_to_output: Bool`` applies LoRA to the model's final output projection.
  This is usually a projection to vocabulary space (e.g. in language models), but
  other modelling tasks may have different projections - classifier models will project
  to the number of classes, for example

.. note::

  Models which use tied embeddings (such as Gemma and Qwen2 1.5B and 0.5B) for the
  final output projection do not support ``apply_lora_to_output``.

These are all specified under the ``model`` flag or config entry, i.e:

.. code-block:: bash

  tune run lora_finetune_single_device --config llama3/8B_lora_single_device  \
  model.apply_lora_to_mlp=True \
  model.lora_attn_modules=["q_proj","k_proj","v_proj"]

.. code-block:: yaml

  model:
    apply_lora_to_mlp: True
    model.lora_attn_modules: ["q_proj", "k_proj", "v_proj"]

Secondly, parameters which control the scale of the impact of LoRA on the model:

* ``lora_rank: int`` affects the scale of the LoRA decomposition, where ``lora_rank << in_dim`` and ``lora_rank << out_dim``
  \- the dimensions of an arbitrary linear layer in the model. Concretely, ``lora_rank`` reduces the number of gradients stored
  in a linear fashion from ``in_dim * out_dim`` to ``lora_rank * (in_dim + out_dim)``. Typically, we have ``lora_rank in [8, 128]``.
* ``lora_alpha: float`` affects the magnitude of the LoRA updates. A larger alpha results in larger updates to the base model weights
  , potentially at the cost of training stability, conversely, smaller alpha can stabilize training at the cost of slower learning.
  We provide default settings for these parameters which we've tested with all of our models, but we encourage you to adjust them
  to your specific use case. Typically, one jointly changes ``lora_rank`` and ``lora_alpha`` together, where ``lora_alpha ~= 2*lora_rank``.
* ``lora_dropout`` introduces dropout in the LoRA layers to help regularize training. We default to 0.0 for all of our models.

As above, these parameters are also specified under the ``model`` flag or config entry.

.. note::

  To get a deeper sense of how LoRA parameters affect memory usage during training,
  see the :ref:`relevant section in our Llama2 LoRA tutorial<lora_tutorial_memory_tradeoff_label>`.

.. _glossary_qlora:

Quantized Low Rank Adaptation (QLoRA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*What's going on here?*

`QLoRA <https://arxiv.org/abs/2305.14314>`_ is an enhancement on top of `LoRA <https://arxiv.org/abs/2106.09685>`_
that maintains the frozen model parameters from LoRA in 4-bit quantized precision, thereby reducing memory usage.
This is enabled through a novel  4-bit NormalFloat (NF4) data type proposed by the authors, which allows for 4-8x less
parameter memory usage whilst retaining model accuracy. You can read our tutorial on :ref:`finetuning Llama2 with QLoRA<qlora_finetune_label>`
for a deeper understanding of how it works.

When considering using QLoRA to reduce memory usage, it's worth noting that QLoRA prevents accuracy degradation during quantization
by up-casting quantized parameters to the original higher precision datatype during model forward passes - this up-casting may
incur penalties to training speed. The :ref:`relevant section <qlora_compile_label>` in our QLoRA tutorial demonstrates the usage of ``torch.compile``
to address this by speeding up training.

*Sounds great! How do I use it?*

You can finetune using QLoRA with any of our LoRA recipes, i.e. recipes with the ``lora_`` prefix, e.g. :ref:`lora_finetune_single_device<lora_finetune_recipe_label>`. These recipes utilize
QLoRA-enabled model builders, which we support for all our models, and also use the ``qlora_`` prefix, e.g.
the :func:`torchtune.models.llama3.llama3_8b` model has a corresponding :func:`torchtune.models.llama3.qlora_llama3_8b`.
We aim to provide a comprehensive set of configurations to allow you to get started with training with QLoRA quickly,
just specify any config with ``_qlora`` in its name, e.g:


.. code-block:: bash

  tune run lora_finetune_single_device --config llama3/8B_qlora_single_device

All the rest of the LoRA parameters remain the same for QLoRA - check out the section above on :ref:`LoRA <glossary_lora>`
to see how to configure.

.. _glossary_distrib:

.. TODO

.. Distributed
.. -----------

.. .. _glossary_fsdp:

.. Fully Sharded Data Parallel (FSDP)
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. All our ``_distributed`` recipes use `FSDP <https://pytorch.org/docs/stable/fsdp.html>`.
.. .. _glossary_fsdp2:
