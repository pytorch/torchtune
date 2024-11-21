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

   ":ref:`glossary_precision`", "You'll usually want to leave this as its default ``bfloat16``. It uses 2 bytes per model parameter instead of 4 bytes when using ``float32``."
   ":ref:`glossary_act_ckpt`", "Use when you're memory constrained and want to use a larger model, batch size or context length. Be aware that it will slow down training speed."
   ":ref:`glossary_act_off`", "Similar to activation checkpointing, this can be used when memory constrained, but may decrease training speed. This **should** be used alongside activation checkpointing."
   ":ref:`glossary_grad_accm`", "Helpful when memory-constrained to simulate larger batch sizes. Not compatible with optimizer in backward. Use it when you can already fit at least one sample without OOMing, but not enough of them."
   ":ref:`glossary_low_precision_opt`", "Use when you want to reduce the size of the optimizer state. This is relevant when training large models and using optimizers with momentum, like Adam. Note that lower precision optimizers may reduce training stability/accuracy."
   ":ref:`glossary_opt_in_bwd`", "Use it when you have large gradients and can fit a large enough batch size, since this is not compatible with ``gradient_accumulation_steps``."
   ":ref:`glossary_cpu_offload`", "Offloads optimizer states and (optionally) gradients to CPU, and performs optimizer steps on CPU. This can be used to significantly reduce GPU memory usage at the cost of CPU RAM and training speed. Prioritize using it only if the other techniques are not enough."
   ":ref:`glossary_lora`", "When you want to significantly reduce the number of trainable parameters, saving gradient and optimizer memory during training, and significantly speeding up training. This may reduce training accuracy"
   ":ref:`glossary_qlora`", "When you are training a large model, since quantization will save 1.5 bytes * (# of model parameters), at the potential cost of some training speed and accuracy."
   ":ref:`glossary_dora`", "a variant of LoRA that may improve model performance at the cost of slightly more memory."


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

To enable activation checkpointing, use ``enable_activation_checkpointing=True``.

.. _glossary_act_off:

Activation Offloading
---------------------

*What's going on here?*

You may have just read about activation checkpointing! Similar to checkpointing, offloading is a memory
efficiency technique that allows saving GPU VRAM by temporarily moving activations to CPU and bringing
them back when needed in the backward pass.

See `PyTorch autograd hook tutorial <https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html#saving-tensors-to-cpu>`_
for more details about how this is implemented through :func:`torch.autograd.graph.saved_tensors_hooks`.

This setting is especially helpful for larger batch sizes, or longer context lengths when you're memory constrained.
While of course it takes runtime and resources to move Tensors from GPU to CPU and back, the implementation in
torchtune uses multiple CUDA streams (when available) in order to overlap the extra communication with the computation
to hide the extra runtime. As the communication workload is variable depending on the number and size of tensors being
offloaded, we do not recommend using it unless :ref:`glossary_act_ckpt` is also enabled, in which case only the checkpointed
tensors will be offloaded.

*Sounds great! How do I use it?*

To enable activation offloading, use the ``enable_activation_offloading`` config entry or flag
in our lora finetuning single device recipe, e.g. ``enable_activation_offloading=True``. To allow
usage of streams, make sure you are on a torch version equal to or later than PyTorch.

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

Gradient accumulation is especially useful when you can fit at least one sample in your GPU. In this case, artificially increasing the batch by
accumulating gradients might give you faster training speeds than using other memory optimization techniques that trade-off memory for speed, like :ref:`activation checkpointing <glossary_act_ckpt>`.

*Sounds great! How do I use it?*

All of our finetuning recipes support simulating larger batch sizes by accumulating gradients. Just set the
``gradient_accumulation_steps`` flag or config entry.

.. note::

  Gradient accumulation should always be set to 1 when :ref:`fusing the optimizer step into the backward pass <glossary_opt_in_bwd>`.

Optimizers
----------

.. _glossary_low_precision_opt:

Lower Precision Optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^

*What's going on here?*

In addition to :ref:`reducing model and optimizer precision <glossary_precision>` during training, we can further reduce precision in our optimizer states.
All of our recipes support lower-precision optimizers from the `torchao <https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim>`_ library.
For single device recipes, we also support `bitsandbytes <https://huggingface.co/docs/bitsandbytes/main/en/index>`_.

A good place to start might be the :class:`torchao.prototype.low_bit_optim.AdamW8bit` and :class:`bitsandbytes.optim.PagedAdamW8bit` optimizers.
Both reduce memory by quantizing the optimizer state dict. Paged optimizers will also offload to CPU if there isn't enough GPU memory available. In practice,
you can expect higher memory savings from bnb's PagedAdamW8bit but higher training speed from torchao's AdamW8bit.

*Sounds great! How do I use it?*

To use this in your recipes, make sure you have installed torchao (``pip install torchao``) or bitsandbytes (``pip install bitsandbytes``). Then, enable
a low precision optimizer using the :ref:`cli_label`:


.. code-block:: bash

  tune run <RECIPE> --config <CONFIG> \
  optimizer=torchao.prototype.low_bit_optim.AdamW8bit

.. code-block:: bash

  tune run <RECIPE> --config <CONFIG> \
  optimizer=bitsandbytes.optim.PagedAdamW8bit

or by directly :ref:`modifying a config file<config_tutorial_label>`:

.. code-block:: yaml

  optimizer:
    _component_: bitsandbytes.optim.PagedAdamW8bit
    lr: 2e-5

.. _glossary_opt_in_bwd:

Fusing Optimizer Step into Backward Pass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

In torchtune, you can enable this feature using the ``optimizer_in_bwd`` flag. This feature works best when using a stateful optimizer
with a model with a lot of parameters, and when you don't need to use :ref:`gradient accumulation <glossary_grad_accm>`.
You won't see meaningful impact when finetuning LoRA recipes, since in this case the number of parameters being updated are small.

.. _glossary_cpu_offload:

Offloading Optimizer/Gradient states to CPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*What's going on here?*

We've mentioned above the concept of optimizer states - memory used by the stateful optimizers to maintain a state of gradient statistics, and
model gradients - tensors used to store gradients when we perform model backwards passes. We support using CPU offloading in our single-device recipes
through the `CPUOffloadOptimizer <https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#optimizer-cpu-offload>`_ from ``torchao``.

This optimizer can wrap any base optimizer and works by keeping the optimizer states and performing the optimizer step on CPU, thus reducing
GPU memory usage by the size of the optimizer states. Additionally, we can also offload gradients to the CPU by using `offload_gradients=True`.

If finetuning on a single-device, another option is to use the ``PagedAdamW8bit`` from bitsandbytes, mentioned :ref:`above <glossary_low_precision_opt>`, which will *only* offload to CPU
when there is not enough GPU available.

*Sounds great! How do I use it?*

To use this optimizer in your recipes, set the ``optimizer`` key in your config to :class:`torchao.prototype.low_bit_optim.CPUOffloadOptimizer`, which
will use the :class:`torch.optim.AdamW` optimizer with ``fused=True`` as the base optimizer. For example, to use this optimizer to offload
both optimizer states and gradients to CPU:

.. code-block:: bash

  tune run <RECIPE> --config <CONFIG> \
  optimizer=optimizer=torchao.prototype.low_bit_optim.CPUOffloadOptimizer \
  optimizer.offload_gradients=True \
  lr=4e-5


or by directly :ref:`modifying a config file<config_tutorial_label>`:

.. code-block:: yaml

  optimizer:
    _component_: torchao.prototype.low_bit_optim.CPUOffloadOptimizer
    offload_gradients: True
    # additional key-word arguments can be passed to torch.optim.AdamW
    lr: 4e-5

or using it directly in your code, which allows you to change the base optimizer:

.. code-block:: python

 from torchao.prototype.low_bit_optim import CPUOffloadOptimizer
 from torch.optim import Adam

 optimizer = CPUOffloadOptimizer(
     model.parameters(), # your model here
     Adam,
     lr=1e-5,
     fused=True
 )

Some helpful hints from the ``torchao`` `CPUOffloadOptimizer page <https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#optimizer-cpu-offload>`_:

* The CPU optimizer step is often the bottleneck when optimizer CPU offload is used. To minimize the slowdown, it is recommended to (1) use full ``bf16`` training so that parameters, gradients, and optimizer states are in ``bf16``; and (2) give GPU more work per optimizer step to amortize the offloading time (e.g. larger batch size with activation checkpointing, gradient accumulation).
* Gradient accumulation should always be set to 1 when ``offload_gradients=True``, as gradients are cleared on GPU every backward pass.
* This optimizer works by keeping a copy of parameters and pre-allocating gradient memory on CPU. Therefore, expect your RAM usage to increase by 4x model size.
* This optimizer is only supported for single-device recipes. To use CPU-offloading in distributed recipes, use ``fsdp_cpu_offload=True`` instead. See :class:`torch.distributed.fsdp.FullyShardedDataParallel` for more details and `FSDP1 vs FSDP2 <https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md>`_ to see how they differ.


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
  model.lora_attn_modules=["q_proj","k_proj","v_proj","output_proj"]

.. code-block:: yaml

  model:
    _component_: torchtune.models.llama3.lora_llama3_8b
    apply_lora_to_mlp: True
    model.lora_attn_modules: ["q_proj", "k_proj", "v_proj","output_proj"]

Secondly, parameters which control the scale of the impact of LoRA on the model:

* ``lora_rank: int`` affects the scale of the LoRA decomposition, where ``lora_rank << in_dim`` and ``lora_rank << out_dim``
  \- the dimensions of an arbitrary linear layer in the model. Concretely, ``lora_rank`` reduces the number of gradients stored
  in a linear fashion from ``in_dim * out_dim`` to ``lora_rank * (in_dim + out_dim)``. Typically, we have ``lora_rank in [8, 256]``.
* ``lora_alpha: float`` affects the magnitude of the LoRA updates. A larger alpha results in larger updates to the base model weights
  , potentially at the cost of training stability, conversely, smaller alpha can stabilize training at the cost of slower learning.
  We provide default settings for these parameters which we've tested with all of our models, but we encourage you to adjust them
  to your specific use case. Typically, one jointly changes ``lora_rank`` and ``lora_alpha`` together, where ``lora_alpha ~= 2*lora_rank``.
* ``lora_dropout`` introduces dropout in the LoRA layers to help regularize training. We default to 0.0 for all of our models.

As above, these parameters are also specified under the ``model`` flag or config entry:

.. code-block:: bash

  tune run lora_finetune_single_device --config llama3/8B_lora_single_device  \
  model.apply_lora_to_mlp=True \
  model.lora_attn_modules=["q_proj","k_proj","v_proj","output_proj"] \
  model.lora_rank=32 \
  model.lora_alpha=64

.. code-block:: yaml

  model:
    _component_: torchtune.models.llama3.lora_llama3_8b
    apply_lora_to_mlp: True
    lora_attn_modules: ["q_proj", "k_proj", "v_proj","output_proj"]
    lora_rank: 32
    lora_alpha: 64

.. note::

  To get a deeper sense of how LoRA parameters affect memory usage during training,
  see the :ref:`relevant section in our Llama2 LoRA tutorial<lora_tutorial_memory_tradeoff_label>`.

.. _glossary_qlora:

Quantized Low Rank Adaptation (QLoRA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*What's going on here?*

`QLoRA <https://arxiv.org/abs/2305.14314>`_ is a memory enhancement on top of `LoRA <https://arxiv.org/abs/2106.09685>`_
that maintains the frozen model parameters from LoRA in 4-bit quantized precision, thereby reducing memory usage.
This is enabled through a novel  4-bit NormalFloat (NF4) data type proposed by the authors, which allows for 4-8x less
parameter memory usage whilst retaining model accuracy. You can read our tutorial on :ref:`finetuning Llama2 with QLoRA<qlora_finetune_label>`
for a deeper understanding of how it works.

When considering using QLoRA to reduce memory usage, it's worth noting that QLoRA is slower than LoRA and may not be worth it if
the model you are finetuning is small. In numbers, QLoRA saves roughly 1.5 bytes * (# of model parameters). Also, although QLoRA quantizes the model,
it minimizes accuracy degradation by up-casting quantized parameters to the original higher precision datatype during model forward passes - this up-casting may incur penalties to training speed.
The :ref:`relevant section <qlora_compile_label>` in our QLoRA tutorial demonstrates the usage of ``torch.compile`` to address this by speeding up training.

*Sounds great! How do I use it?*

You can finetune using QLoRA with any of our LoRA recipes, i.e. recipes with the ``lora_`` prefix, e.g. :ref:`lora_finetune_single_device<lora_finetune_recipe_label>`. These recipes utilize
QLoRA-enabled model builders, which we support for all our models, and also use the ``qlora_`` prefix, e.g.
the :func:`torchtune.models.llama3.llama3_8b` model has a corresponding :func:`torchtune.models.llama3.qlora_llama3_8b`.
We aim to provide a comprehensive set of configurations to allow you to get started with training with QLoRA quickly,
just specify any config with ``_qlora`` in its name.

All the rest of the LoRA parameters remain the same for QLoRA - check out the section above on :ref:`LoRA <glossary_lora>`
to see how to configure these parameters.

To configure from the command line:

.. code-block:: bash

  tune run lora_finetune_single_device --config llama3/8B_qlora_single_device \
  model.apply_lora_to_mlp=True \
  model.lora_attn_modules=["q_proj","k_proj","v_proj"] \
  model.lora_rank=32 \
  model.lora_alpha=64


or, by modifying a config:

.. code-block:: yaml

  model:
    _component_: torchtune.models.qlora_llama3_8b
    apply_lora_to_mlp: True
    lora_attn_modules: ["q_proj", "k_proj", "v_proj"]
    lora_rank: 32
    lora_alpha: 64

.. _glossary_dora:

Weight-Decomposed Low-Rank Adaptation (DoRA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*What's going on here?*

`DoRA <https://arxiv.org/abs/2402.09353>`_ is another PEFT technique which builds on-top of LoRA by
further decomposing the pre-trained weights into two components: magnitude and direction. The magnitude component
is a scalar vector that adjusts the scale, while the direction component corresponds to the original LoRA decomposition and
updates the orientation of weights.

DoRA adds a small overhead to LoRA training due to the addition of the magnitude parameter, but it has been shown to
improve the performance of LoRA, particularly at low ranks.

*Sounds great! How do I use it?*

Much like LoRA and QLoRA, you can finetune using DoRA with any of our LoRA recipes. We use the same model builders for LoRA
as we do for DoRA, so you can use the ``lora_`` version of any model builder with ``use_dora=True``. For example, to finetune
:func:`torchtune.models.llama3.llama3_8b` with DoRA, you would use :func:`torchtune.models.llama3.lora_llama3_8b` with ``use_dora=True``:

.. code-block:: bash

  tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  model.use_dora=True

.. code-block:: yaml

  model:
    _component_: torchtune.models.lora_llama3_8b
    use_dora: True

Since DoRA extends LoRA, the parameters for :ref:`customizing LoRA <glossary_lora>` are identical. You can also quantize the base model weights like in :ref:`glossary_qlora` by using ``quantize=True`` to reap
even more memory savings!

.. code-block:: bash

  tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  model.apply_lora_to_mlp=True \
  model.lora_attn_modules=["q_proj","k_proj","v_proj"] \
  model.lora_rank=16 \
  model.lora_alpha=32 \
  model.use_dora=True \
  model.quantize_base=True

.. code-block:: yaml

  model:
    _component_: torchtune.models.lora_llama3_8b
    apply_lora_to_mlp: True
    lora_attn_modules: ["q_proj", "k_proj", "v_proj"]
    lora_rank: 16
    lora_alpha: 32
    use_dora: True
    quantize_base: True


.. note::

   Under the hood, we've enabled DoRA by adding the :class:`~torchtune.modules.peft.DoRALinear` module, which we swap
   out for :class:`~torchtune.modules.peft.LoRALinear` when ``use_dora=True``.

.. _glossary_distrib:


.. TODO

.. Distributed
.. -----------

.. .. _glossary_fsdp:

.. Fully Sharded Data Parallel (FSDP)
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. All our ``_distributed`` recipes use `FSDP <https://pytorch.org/docs/stable/fsdp.html>`.
.. .. _glossary_fsdp2:
