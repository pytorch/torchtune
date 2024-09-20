.. _lora_finetune_label:

============================
Fine-Tuning Llama2 with LoRA
============================

This guide will teach you about `LoRA <https://arxiv.org/abs/2106.09685>`_, a parameter-efficient finetuning technique,
and show you how you can use torchtune to finetune a Llama2 model with LoRA.
If you already know what LoRA is and want to get straight to running
your own LoRA finetune in torchtune, you can jump to :ref:`LoRA finetuning recipe in torchtune<lora_recipe_label>`.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * What LoRA is and how it saves memory during finetuning
      * An overview of LoRA components in torchtune
      * How to run a LoRA finetune using torchtune
      * How to experiment with different LoRA configurations

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with :ref:`torchtune<overview_label>`
      * Make sure to :ref:`install torchtune<install_label>`
      * Make sure you have downloaded the :ref:`Llama2-7B model weights<download_llama_label>`

What is LoRA?
-------------

`LoRA <https://arxiv.org/abs/2106.09685>`_ is an adapter-based method for
parameter-efficient finetuning that adds trainable low-rank decomposition matrices to different layers of a neural network,
then freezes the network's remaining parameters. LoRA is most commonly applied to
transformer models, in which case it is common to add the low-rank matrices
to some of the linear projections in each transformer layer's self-attention.

.. note::

    If you're unfamiliar, check out these references for the `definition of rank <https://en.wikipedia.org/wiki/Rank_(linear_algebra)>`_
    and discussion of `low-rank approximations <https://en.wikipedia.org/wiki/Low-rank_approximation>`_.

By finetuning with LoRA (as opposed to finetuning all model parameters),
you can expect to see memory savings due to a substantial reduction in the
number of parameters with gradients. When using an optimizer with momentum,
like `AdamW <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>`_,
you can expect to see further memory savings from the optimizer state.

.. note::

    LoRA memory savings come primarily from gradient and optimizer states,
    so if your model's peak memory comes in its :code:`forward()` method, then LoRA
    may not reduce peak memory.

How does LoRA work?
-------------------

LoRA replaces weight update matrices with a low-rank approximation. In general, weight updates
for an arbitrary :code:`nn.Linear(in_dim,out_dim)` layer could have rank as high as
:code:`min(in_dim,out_dim)`. LoRA (and other related papers such as `Aghajanyan et al. <https://arxiv.org/abs/2012.13255>`_)
hypothesize that the `intrinsic dimension <https://en.wikipedia.org/wiki/Intrinsic_dimension>`_
of these updates during LLM fine-tuning can in fact be much lower.
To take advantage of this property, LoRA finetuning will freeze the original model,
then add a trainable weight update from a low-rank projection. More explicitly, LoRA trains two
matrices :code:`A` and :code:`B`. :code:`A` projects the inputs down to a much smaller rank (often four or eight in practice), and
:code:`B` projects back up to the dimension output by the original linear layer.

The image below gives a simplified representation of a single weight update step from a full finetune
(on the left) compared to a weight update step with LoRA (on the right). The LoRA matrices :code:`A` and :code:`B`
serve as an approximation to the full rank weight update in blue.

.. image:: /_static/img/lora_diagram.png

Although LoRA introduces a few extra parameters in the model :code:`forward()`, only the :code:`A` and :code:`B` matrices are trainable.
This means that with a rank :code:`r` LoRA decomposition, the number of gradients we need to store reduces
from :code:`in_dim*out_dim` to :code:`r*(in_dim+out_dim)`. (Remember that in general :code:`r`
is much smaller than :code:`in_dim` and :code:`out_dim`.)

For example, in the 7B Llama2's self-attention, :code:`in_dim=out_dim=4096` for the Q, K,
and V projections. This means a LoRA decomposition of rank :code:`r=8` will reduce the number of trainable
parameters for a given projection from :math:`4096 * 4096 \approx 15M` to :math:`8 * 8192 \approx 65K`, a
reduction of over 99%.

Let's take a look at a minimal implementation of LoRA in native PyTorch.


.. code-block:: python

  import torch
  from torch import nn

  class LoRALinear(nn.Module):
    def __init__(
      self,
      in_dim: int,
      out_dim: int,
      rank: int,
      alpha: float,
      dropout: float
    ):
      # These are the weights from the original pretrained model
      self.linear = nn.Linear(in_dim, out_dim, bias=False)

      # These are the new LoRA params. In general rank << in_dim, out_dim
      self.lora_a = nn.Linear(in_dim, rank, bias=False)
      self.lora_b = nn.Linear(rank, out_dim, bias=False)

      # Rank and alpha are commonly-tuned hyperparameters
      self.rank = rank
      self.alpha = alpha

      # Most implementations also include some dropout
      self.dropout = nn.Dropout(p=dropout)

      # The original params are frozen, and only LoRA params are trainable.
      self.linear.weight.requires_grad = False
      self.lora_a.weight.requires_grad = True
      self.lora_b.weight.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      # This would be the output of the original model
      frozen_out = self.linear(x)

      # lora_a projects inputs down to the much smaller self.rank,
      # then lora_b projects back up to the output dimension
      lora_out = self.lora_b(self.lora_a(self.dropout(x)))

      # Finally, scale by the alpha parameter (normalized by rank)
      # and add to the original model's outputs
      return frozen_out + (self.alpha / self.rank) * lora_out

There are some other details around initialization which we omit here, but if you'd like to know more
you can see our implementation in :class:`~torchtune.modules.peft.LoRALinear`.
Now that we understand what LoRA is doing, let's look at how we can apply it to our favorite models.

Applying LoRA to Llama2 models
------------------------------

With torchtune, we can easily apply LoRA to Llama2 with a variety of different configurations.
Let's take a look at how to construct Llama2 models in torchtune with and without LoRA.

.. code-block:: python

  from torchtune.models.llama2 import llama2_7b, lora_llama2_7b

  # Build Llama2 without any LoRA layers
  base_model = llama2_7b()

  # The default settings for lora_llama2_7b will match those for llama2_7b
  # We just need to define which layers we want LoRA applied to.
  # Within each self-attention, we can choose from ["q_proj", "k_proj", "v_proj", and "output_proj"].
  # We can also set apply_lora_to_mlp=True or apply_lora_to_output=True to apply LoRA to other linear
  # layers outside of the self-attention.
  lora_model = lora_llama2_7b(lora_attn_modules=["q_proj", "v_proj"])

.. note::

    Calling :func:`lora_llama_2_7b <torchtune.models.llama2.lora_llama2_7b>` alone will not handle the definition of which parameters are trainable.
    See :ref:`below<setting_trainable_params>` for how to do this.

Let's inspect each of these models a bit more closely.

.. code-block:: bash

  # Print the first layer's self-attention in the usual Llama2 model
  >>> print(base_model.layers[0].attn)
  MultiHeadAttention(
    (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (output_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (pos_embeddings): RotaryPositionalEmbeddings()
  )

  # Print the same for Llama2 with LoRA weights
  >>> print(lora_model.layers[0].attn)
  MultiHeadAttention(
    (q_proj): LoRALinear(
      (dropout): Dropout(p=0.0, inplace=False)
      (lora_a): Linear(in_features=4096, out_features=8, bias=False)
      (lora_b): Linear(in_features=8, out_features=4096, bias=False)
    )
    (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (v_proj): LoRALinear(
      (dropout): Dropout(p=0.0, inplace=False)
      (lora_a): Linear(in_features=4096, out_features=8, bias=False)
      (lora_b): Linear(in_features=8, out_features=4096, bias=False)
    )
    (output_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (pos_embeddings): RotaryPositionalEmbeddings()
  )


Notice that our LoRA model's layer contains additional weights in the Q and V projections,
as expected. Additionally, inspecting the type of :code:`lora_model` and
:code:`base_model`, would show that they are both instances of the same :class:`~torchtune.modules.TransformerDecoder`.
(Feel free to verify this for yourself.)

Why does this matter? torchtune makes it easy to load checkpoints for LoRA directly from our Llama2
model without any wrappers or custom checkpoint conversion logic.

.. code-block:: python

  # Assuming that base_model already has the pretrained Llama2 weights,
  # this will directly load them into your LoRA model without any conversion necessary.
  lora_model.load_state_dict(base_model.state_dict(), strict=False)

.. note::
    Whenever loading weights with :code:`strict=False`, you should verify that any missing or extra keys in
    the loaded :code:`state_dict` are as expected. torchtune's LoRA recipes do this by default via e.g.
    :func:`validate_state_dict_for_lora() <torchtune.modules.peft.validate_state_dict_for_lora>` or
    :func:`validate_missing_and_unexpected_for_lora() <torchtune.modules.peft.validate_missing_and_unexpected_for_lora>`.

Once we've loaded the base model weights, we also want to set only LoRA parameters to trainable.

.. _setting_trainable_params:

.. code-block:: python

  from torchtune.modules.peft.peft_utils import get_adapter_params, set_trainable_params

  # Fetch all params from the model that are associated with LoRA.
  lora_params = get_adapter_params(lora_model)

  # Set requires_grad=True on lora_params, and requires_grad=False on all others.
  set_trainable_params(lora_model, lora_params)

  # Print the total number of parameters
  total_params = sum([p.numel() for p in lora_model.parameters()])
  trainable_params = sum([p.numel() for p in lora_model.parameters() if p.requires_grad])
  print(
    f"""
    {total_params} total params,
    {trainable_params}" trainable params,
    {(100.0 * trainable_params / total_params):.2f}% of all params are trainable.
    """
  )

  6742609920 total params,
  4194304 trainable params,
  0.06% of all params are trainable.

.. note::
    If you are directly using the LoRA recipe (as detailed :ref:`here<lora_recipe_label>`), you need only pass the
    relevant checkpoint path. Loading model weights and setting trainable parameters will be taken care
    of in the recipe.


.. _lora_recipe_label:

LoRA finetuning recipe in torchtune
-----------------------------------

Finally, we can put it all together and finetune a model using torchtune's `LoRA recipe <https://github.com/pytorch/torchtune/blob/48626d19d2108f92c749411fbd5f0ff140023a25/recipes/lora_finetune.py>`_.
Make sure that you have first downloaded the Llama2 weights and tokenizer by following :ref:`these instructions<download_llama_label>`.
You can then run the following command to perform a LoRA finetune of Llama2-7B with two GPUs (each having VRAM of at least 16GB):

.. code-block:: bash

    tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed --config llama2/7B_lora

.. note::
    Make sure to point to the location of your Llama2 weights and tokenizer. This can be done
    either by adding :code:`checkpointer.checkpoint_files=[my_model_checkpoint_path] tokenizer_checkpoint=my_tokenizer_checkpoint_path`
    or by directly modifying the :code:`7B_lora.yaml` file. See our "":ref:`config_tutorial_label`" recipe
    for more details on how you can easily clone and modify torchtune configs.

.. note::
    You can modify the value of :code:`nproc_per_node` depending on (a) the number of GPUs you have available,
    and (b) the memory constraints of your hardware.

The preceding command will run a LoRA finetune with torchtune's factory settings, but we may want to experiment a bit.
Let's take a closer look at some of the :code:`lora_finetune_distributed` config.

.. code-block:: yaml

  # Model Arguments
  model:
    _component_: lora_llama2_7b
    lora_attn_modules: ['q_proj', 'v_proj']
    lora_rank: 8
    lora_alpha: 16
  ...

We see that the default is to apply LoRA to Q and V projections with a rank of 8.
Some experiments with LoRA have found that it can be beneficial to apply LoRA to all linear layers in
the self-attention, and to increase the rank to 16 or 32. Note that this is likely to increase our max memory,
but as long as we keep :code:`rank<<embed_dim`, the impact should be relatively minor.

Let's run this experiment. We can also increase alpha (in general it is good practice to scale alpha and rank together).

.. code-block:: bash

    tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed --config llama2/7B_lora \
    lora_attn_modules=['q_proj','k_proj','v_proj','output_proj'] \
    lora_rank=32 lora_alpha=64 output_dir=./lora_experiment_1

A comparison of the (smoothed) loss curves between this run and our baseline over the first 500 steps can be seen below.

.. image:: /_static/img/lora_experiment_loss_curves.png

.. note::
    The above figure was generated with W&B. You can use torchtune's :class:`~torchtune.training.metric_logging.WandBLogger`
    to generate similar loss curves, but you will need to install W&B and setup an account separately. For more details on
    using W&B in torchtune, see our ":ref:`wandb_logging`" recipe.

.. _lora_tutorial_memory_tradeoff_label:

Trading off memory and model performance with LoRA
--------------------------------------------------

In the preceding example, we ran LoRA on two devices. But given LoRA's low memory footprint, we can run fine-tuning
on a single device using most commodity GPUs which support `bfloat16 <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#bfloat16_floating-point_format>`_
floating-point format. This can be done via the command:

.. code-block:: bash

    tune run lora_finetune_single_device --config llama2/7B_lora_single_device

On a single device, we may need to be more cognizant of our peak memory. Let's run a few experiments
to see our peak memory during a finetune. We will experiment along two axes:
first, which model layers have LoRA applied, and second, the rank of each LoRA layer. (We will scale
alpha in parallel to LoRA rank, as discussed above.)

To compare the results of our experiments, we can evaluate our models on `truthfulqa_mc2 <https://github.com/sylinrl/TruthfulQA>`_, a task from
the `TruthfulQA <https://arxiv.org/abs/2109.07958>`_ benchmark for language models. For more details on how to run this and other evaluation tasks
with torchtune's EleutherAI evaluation harness integration, see our :ref:`End-to-End Workflow Tutorial <eval_harness_label>`.

Previously, we only enabled LoRA for the linear layers in each self-attention module, but in fact there are other linear
layers we can apply LoRA to: MLP layers and our model's final output projection. Note that for Llama-2-7B the final output
projection maps to the vocabulary dimension (32000 instead of 4096 as in the other linear layers), so enabling LoRA for this layer will increase
our peak memory a bit more than the other layers. We can make the following changes to our config:

.. code-block:: yaml

  # Model Arguments
  model:
    _component_: lora_llama2_7b
    lora_attn_modules: ['q_proj', 'k_proj', 'v_proj', 'output_proj']
    apply_lora_to_mlp: True
    apply_lora_to_output: True
  ...

.. note::
    All the finetuning runs below use the `llama2/7B_lora_single_device <https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_lora_single_device.yaml>`_
    config, which has a default batch size of 2. Modifying the batch size (or other hyperparameters, e.g. the optimizer) will impact both peak memory
    and final evaluation results.

.. list-table::
   :widths: 25 25 25 25 25
   :header-rows: 1

   * - LoRA Layers
     - Rank
     - Alpha
     - Peak Memory
     - Accuracy (truthfulqa_mc2)
   * - Q and V only
     - 8
     - 16
     - **15.57 GB**
     - 0.475
   * - all layers
     - 8
     - 16
     - 15.87 GB
     - 0.508
   * - Q and V only
     - 64
     - 128
     - 15.86 GB
     - 0.504
   * - all layers
     - 64
     - 128
     - 17.04 GB
     - **0.514**

We can see that our baseline settings give the lowest peak memory, but our evaluation performance is relatively lower.
By enabling LoRA for all linear layers and increasing the rank to 64, we see almost a 4% absolute improvement
in our accuracy on this task, but our peak memory also increases by about 1.4GB. These are just a couple simple
experiments; we encourage you to run your own finetunes to find the right tradeoff for your particular setup.

Additionally, if you want to decrease your model's peak memory even further (and still potentially achieve similar
model quality results), you can check out our :ref:`QLoRA tutorial<qlora_finetune_label>`.
