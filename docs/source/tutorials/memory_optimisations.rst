.. _memory_optimisation_overview_label:

============================
Memory Optimisation Overview
============================

**Author**: `Salman Mohammadi <https://github.com/SalmanMohammadi>`_

Torchtune comes with a host of plug-and-play memory optimisation components which give you lots of flexibility
to `tune` our recipes to your hardware. This page provides a brief glossary of these components and how you might use them.


.. _glossary_precision:

Model Precision
---------------

Model precision is controlled using the ``dtype`` flag or config entry in all our recipes. When ``dtype=bf16``,
all activations, gradients and optimizer states are in bfloat16. In
most cases this should halve the memory footprint of full precision (fp32) training, without
loss in model quality (will depend on the model, training data and other settings). For
GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
precision are currently not supported.

.. _glossary_act_ckpt:

Activation Checkpointing
------------------------

Activation checkpointing can be controlled using the ``activation_checkpointing``
flag or config entry in all our finetuning recipes. Enabling activation checkpointing helps reduce the
memory footprint since we no longer keep activations in memory and instead recompute them during the backward pass.
This is especially helpful for larger batch sizes when you're memory constrained. However, these savings in memory
come at the cost of training performance, and in most cases training can slow-down quite a bit as
a result of this activation recomputation.

.. _glossary_grad_accm:

Gradient Accumulation
---------------------

All of our finetuning recipes support simulating larger batch sizes by accumulating gradients. This is
controlled using the ``gradient_accumulation_steps`` flag or config entry. The
total number of samples used for a gradient update is:

  ``total_batch_size = batch_size * gradient_accumulation_steps``

For example: with ``batch_size=1`` and ``gradient_accumulation_steps=32`` we get a total batch size of 32.

Gradient accumulation is especially useful when you are memory constrained. In this case,
accumulating gradients might give you better training speed than enabling activation
checkpointing, since activation checkpointing reduces memory consumption at the cost of repeated
computations.

.. note::

  Gradient accumulation should always be set to 1 when using :ref:`fusing the optimizer step into the backward pass <glossary_opt_in_bwd>`.

.. _glossary_low_precision_opt:

Lower Precision Optimizers
--------------------------

In addition to reducing model precision during training, we can also reduce precision in our optimzer state.
All of our fine-tuning recipes support lower-sprecision optimizers from the `bitsandbytes <https://huggingface.co/docs/bitsandbytes/main/en/index>`_ library -
a good place to start might be the ``Adam8bit`` and ``PagedAdamW`` optimizers, which we've tested our recipes with.

To use this in your recipes, make sure you have installed bitsandbytes (``pip install bitsandbytes``). Then, enable
a low precision optimizer using the :ref:`cli_label`:

.. code-block:: bash

  tune run <RECIPE> --config <CONFIG> \
  optimizer._component_=bitsandbytes.optim.PagedAdamW8bit

or by directly :ref:`modify a config file<config_tutorial_label>`:

.. code-block:: yaml

  optimizer:
    _component_: bitsandbytes.optim.PagedAdamW
    lr: 2e-5

.. note::

  Utilising lower-precision optimizers as a memory-saving technique works best when using an optimizer which
  maintain a state of gradient statistics. As an alternative, you could try a stateless optimizer
  such as `stochastic gradient descent <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_
  without momentum, which doesn't require any additional memory usage, and may result in lower memory usage
  than a low-precision "stateful" optimizer. This concept is also important in :ref:`fusing the optimizer step
  into the backward pass<glossary_opt_in_bwd>`.

.. _glossary_opt_in_bwd:

Fusing Optimizer Step into Backward Pass
----------------------------------------

We previously noted the distinction between stateful and stateless optimzers. As an alternative to lower-precision optimizers, let's
consider a technique which enables the use of "stateful" optimizers such as ``AdamW``, without the memory overhead of gradient statistics,
by completely removing the buffer of gradients which are stored by the optimizer during its ``step()``.

We encourage you to read through the fantastic PyTorch tutorial on this concept:
`*How to save memory by fusing the optimizer step into the backward pass* <https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html>`_

.. todo (SalmanMohammadi) ref full finetune

You can enable this feature using the ``optimizer_in_bwd`` flag, which is currently only supported in our
full finetune recipe. You might want to use this feature when:

* When gradient memory is particularly large i.e. when using a stateful optimizer.
* When you don't need gradient accumulation.

.. note::

  You'll need to ensure you have PyTorch ``2.1.0`` or later to use this feature. See the PyTorch install instructions
  `here <https://pytorch.org/get-started/locally/>`_.

.. _glossary_peft:

Parameter Efficient Fine-Tuning (PEFT)
--------------------------------------

.. _glossary_lora:

Low Rank Adaptation (LoRA)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Our tutorial on :ref:`finetuning Llama2 with LoRA<lora_finetune_label>` does a fantastic job of explaining LoRA, and how to use it. Here's
an excerpt to give you a quick idea of how it works:

  `LoRA <https://arxiv.org/abs/2106.09685>`_ is an adapter-based method for
  parameter-efficient finetuning that adds trainable low-rank decomposition matrices to different layers of a neural network,
  then freezes the network's remaining parameters. LoRA is most commonly applied to
  transformer models, in which case it is common to add the low-rank matrices
  to some of the linear projections in each transformer layer's self-attention.

You can fine-tune with LoRA with all of our models, using any of  our ``_lora`` recipes. Just add the ``lora_`` prefix to the
name of any model you're interested in. By using the :ref:`cli_label`:

.. code-block:: bash

  tune run <RECIPE> --config <CONFIG> \
  model._component_=torchtune.models.<model>.lora_<model>

For example, use ``lora_llama2_7b`` instead of ``llama2_7b``, ``lora_gemma_2b`` instead of ``gemma_2b``, etc.
You can also directly :ref:`modify a config file<config_tutorial_label>`:

.. code-block:: yaml

  model:
    _component_: torchtune.models.<model>.lora_<model>

There are two sets of parameters to customize LoRA to suit your needs. Firstly, the parameters which control
which linear layers LoRA should be applied to in the model:

* ``lora_attn_modules: List[str]`` accepts a list of strings specifying which layers of the model to apply
  LoRA to:

  * ``q_proj`` applies LoRA to the query projection layer.
  * ``k_proj`` applies LoRA to the key projection layer.
  * ``v_proj`` applies LoRA to the value projection layer.
  * ``output_proj`` applies LoRA to the attention output projection layer.

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

  tune run <RECIPE> --config <CONFIG> \
  model.apply_lora_to_mlp \
  model.lora_attn_modules=["q_proj", "k_proj", "v_proj"]

.. code-block:: yaml

  model:
    apply_lora_to_mlp: True
    model.lora_attn_modules: ["q_proj", "k_proj", "v_proj"]

Secondly, parameters which control the scale of the impact of LoRA on the model:

* ``lora_rank: int`` affects the scale of the LoRA decomposition, where ``lora_rank << in_dim`` and ``lora_rank << out_dim``
  \- the dimensions of an arbitrary linear layer in the model. Concretely, ``lora_rank`` reduces the number of gradients stored
  in a linear fashion from ``in_dim * out_dim`` to ``lora_rank * (in_dim + out_dim)`` -
* ``lora_alpha: float`` affects the magnitude of the LoRA updates. A larger alpha results in larger updates to the base model weights
  , potentially at the cost of training stability, conversely, smaller alpha can stabilize training at the cost of slower learning.
  We provide default settings for these parameters which we've tested with all of our models, but we encourage you to adjust them
  to your specific use case. Typically, one jointly changes ``lora_rank`` and ``lora_alpha`` together.
* ``lora_dropout`` introduces dropout in the LoRA layers to help regularize training. We default to 0.0 for all of our models.

As above, these parameters are also specified under the ``model`` flag or config entry.

.. note::

  To get a deeper sense of how LoRA parameters affect memory usage during training,
  see the :ref:`relevant section in our Llama2 LoRA tutorial<lora_tutorial_memory_tradeoff_label>`.


Quantized Low Rank Adaptation (LoRA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our tutorial on :ref:`finetuning Llama2 with QLoRA<qlora_finetune_label>` does a fantastic job of explaining QLoRA, and how to use it. Here's
an excerpt to give you a quick idea of how it works:

  `QLoRA <https://arxiv.org/abs/2305.14314>`_ is an enhancement on top of `LoRA <https://arxiv.org/abs/2106.09685>`_
  that maintains the frozen model parameters from LoRA in 4-bit quantized precision, thereby reducing memory usage.

Just like LoRA, you can fine-tune with QLoRA with all of our models, using any of  our ``_lora`` recipes. Just add the ``qlora_`` prefix to the
name of any model you're interested in. To avoid repetition, please refer to the section above for how to
configure this in your recipes. All the rest of the LoRA parameters remain the same for QLoRA.

.. _glossary_distrib:

Distributed
-----------

.. _glossary_fsdp:

Fully Sharded Data Parallel (FSDP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All our ``_distributed`` recipes use `FSDP <https://pytorch.org/docs/stable/fsdp.html>`.
.. _glossary_fsdp2:

(Experimental) Fully Sharded Data Parallel 2 (FSDP2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This directory contains distributed training recipes for LoRA and QLoRA using `FSDP2 <https://github.com/pytorch/pytorch/issues/114299>`_.
Currently FSDP2 is only available in PyTorch nightly releases.
