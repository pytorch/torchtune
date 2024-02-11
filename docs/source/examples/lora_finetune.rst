.. _lora_finetune_label:

===========================
Finetuning Llama2 with LoRA
===========================

This guide will teach you about `LoRA <https://arxiv.org/abs/2106.09685>`_, a parameter-efficient finetuning technique,
and show you how you can use TorchTune to finetune a Llama2 model with LoRA.
If you already know what LoRA is and want to get straight to running
your own LoRA finetune in TorchTune, you can jump to :ref:`this section<lora_recipe_label>`.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * What LoRA is and how it saves memory during finetuning
      * An overview of LoRA components in TorchTune
      * How to run a LoRA finetune using TorchTune
      * How to experiment with different LoRA configurations

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with the :ref:`overview of TorchTune<overview_label>`
      * Make sure to :ref:`install TorchTune<install_label>`
      * Make sure you have downloaded the :ref:`Llama2-7B model weights<download_llama_label>`

What is LoRA?
-------------

`LoRA <https://arxiv.org/abs/2106.09685>`_ is a parameter-efficient finetuning technique that adds a trainable
low-rank decomposition to different layers of a neural network, then freezes
the network's remaining parameters. LoRA is most commonly applied to
transformer models, in which case it is common to add the low-rank matrices
to some of the self-attention projections in each transformer layer.

By finetuning with LoRA (as opposed to finetuning all model parameters),
you can expect to see memory savings due to a substantial reduction in the
number of gradient parameters. When using an optimizer with momentum,
like `AdamW <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>`_,
you can expect to see further memory savings from the optimizer state.

.. note::

    LoRA memory savings come primarily from gradient and optimizer states,
    and so if your model's peak memory comes in its :code:`forward()`, then LoRA
    may not reduce peak memory.

How does LoRA work?
-------------------

LoRA replaces weight update matrices with a low-rank approximation. In general, weight updates
for a given linear layer mapping dimension :code:`in_dim` to dimension :code:`out_dim` can have rank as high as
:code:`min(in_dim,out_dim)`. LoRA (and other related papers such as `Aghajanyan et al. <https://arxiv.org/abs/2012.13255>`_)
hypothesize that the `intrinsic dimension <https://en.wikipedia.org/wiki/Intrinsic_dimension>`_
of these updates during LLM fine-tuning can in fact be much lower.
To take advantage of this property, LoRA finetuning will freeze the original model,
then add a trainable weight update from a low-rank projection. More explicitly, LoRA trains two
matrices :code:`A` and :code:`B`. :code:`A` projects the inputs down to a much smaller rank (often four or eight in practice), and
:code:`B` projects back up to the dimension output by the original linear layer.

Although this introduces a few extra parameters in the model :code:`forward()`, only the LoRA matrices are trainable.
This means that with a rank :code:`r` LoRA decomposition, the number of gradients we need to store reduces
from :code:`in_dim*out_dim` to :code:`r*(in_dim+out_dim)`. (Remember that in general :code:`r`
is much smaller than :code:`in_dim` and :code:`out_dim`.)

For example, in the 7B Llama2's self-attention, :code:`in_dim=out_dim=4096` for the Q, K,
and V projections. This means a LoRA decomposition of rank :code:`r=8` will reduce the number of trainable
parameters for a given projection from :math:`4096 * 4096 \approx 15M` to :math:`8 * 8192 \approx 65K`, a
reduction of over 99%.

Let's take a look at a minimal implementation of LoRA in native PyTorch.


.. code-block:: python

  from torch import nn, Tensor

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

    def forward(self, x: Tensor) -> Tensor:
      # This would be the output of the original model
      frozen_out = self.linear(x)

      # lora_a projects inputs down to the much smaller self.rank,
      # then lora_b projects back up to the output dimension
      lora_out = self.lora_b(self.lora_a(self.dropout(x)))

      # Finally, scale by the alpha parameter (normalized by rank)
      # and add to the original model's outputs
      return frozen_out + (self.alpha / self.rank) * lora_out

There are some other details around initialization which we omit here, but otherwise that's
pretty much it. Now that we understand what LoRA is doing, let's look at how we can apply it
to our favorite models.

Applying LoRA to Llama2 models
------------------------------

With TorchTune, we can easily apply LoRA to Llama2 with a variety of different configurations.
Let's take a look at how to construct Llama2 models in TorchTune with and without LoRA.

.. code-block:: python

  from torchtune.models import llama2_7b, lora_llama2_7b

  # Build Llama2 without any LoRA layers
  base_model = llama2_7b()

  # The default settings for lora_llama2_7b will match those for llama2_7b
  # We just need to define which layers we want LoRA applied to.
  # We can choose from ["q_proj", "k_proj", "v_proj", and "output_proj"]
  lora_model = lora_llama2_7b(lora_attn_modules=["q_proj", "v_proj"])

.. note::

    Calling :code:`lora_llama_2_7b` alone will not handle the definition of which parameters are trainable.
    See :ref:`below<setting_trainable_params>` for how to do this.

Let's inspect each of these models a bit more closely.

.. code-block:: python

  # Print the first layer's self-attention in the usual Llama2 model
  print(base_model.layers[0].attn)

  CausalSelfAttention(
    (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (output_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (pos_embeddings): RotaryPositionalEmbeddings()
  )

  # Print the same for Llama2 with LoRA weights
  print(lora_model.layers[0].attn)

  CausalSelfAttention(
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

Why does this matter? TorchTune makes it easy to load checkpoints for LoRA directly from our Llama2
model without any wrappers or custom checkpoint conversion logic.

.. code-block:: python

  # Assuming that base_model already has the pretrained Llama2 weights,
  # this will directly load them into your LoRA model without any conversion necessary.
  lora_model.load_state_dict(base_model.state_dict(), strict=False)

.. note::
    Whenever loading weights with :code:`strict=False`, you should verify that any missing or extra keys in
    the loaded :code:`state_dict` are as expected. TorchTune's LoRA recipe does this by default via
    :func:`torchtune.modules.peft.validate_state_dict_for_lora`.

Once we've loaded the base model weights, we also want to set only LoRA parameters to trainable.

.. _setting_trainable_params:

.. code-block:: python

  from torchtune.modules.peft.peft_utils import get_adapter_params, set_trainable_params

  # Fetch all params from the model that are associated with LoRA.
  lora_params = get_adapter_params(lora_model)

  # Set requires_grad=True on lora_params, and requires_grad=False on all others.
  set_trainable_params(lora_model, lora_params)

  # Print the total number of parameters
  total_params = sum([p.numel() for p in lora_model.params()])
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

LoRA finetuning recipe in TorchTune
-----------------------------------

Finally, we can put it all together and finetune a model using TorchTune's `LoRA recipe <https://github.com/pytorch-labs/torchtune/blob/48626d19d2108f92c749411fbd5f0ff140023a25/recipes/lora_finetune.py>`_.
Make sure that you have first downloaded the Llama2 weights and tokenizer by following :ref:`these instructions<download_llama_label>`.
You can then run the following command to perform a LoRA finetune of Llama2-7B using the Alpaca dataset with two GPUs:

.. code-block:: bash

    tune --nnodes 1 --nproc_per_node 2 lora_finetune --config alpaca_llama2_lora_finetune

.. note::
    Make sure to point to the location of your Llama2 weights and tokenizer. This can be done
    either by adding :code:`--override model_checkpoint=my_model_checkpoint_path tokenizer_checkpoint=my_tokenizer_checkpoint_path`
    or by directly modifying the :code:`alpaca_llama2_lora_finetune.yaml` file. See our :ref:`config_tutorial_label`
    for more details on how you can easily clone and modify TorchTune configs.

.. note::
    You can modify the value of :code:`nproc_per_node` depending on (a) the number of GPUs you have available,
    and (b) the memory constraints of your hardware. See `this table <https://github.com/pytorch-labs/torchtune/tree/main?tab=readme-ov-file#finetuning-resource-requirements>`_
    for peak memory of LoRA finetuning in a couple of common hardware setups.

The preceding command will run a LoRA finetune with TorchTune's factory settings, but we may want to experiment a bit.
Let's take a closer look at some of the :code:`alpaca_llama2_lora_finetune` config.

.. code-block:: yaml

  # Model Arguments
  model: lora_llama2_7b
  lora_attn_modules: ['q_proj', 'v_proj']
  lora_rank: 8
  lora_alpha: 16
  ...

We see that the default is to apply LoRA to Q and V projections with a rank of 8.
Some experiments with LoRA have found that it can be beneficial to apply LoRA to all linear layers in
the self-attention, and to increase the rank to 16. Note that this is likely to increase our max memory,
but as long as we keep :code:`rank<<embed_dim`, the impact should be relatively minor.

We can run this experiment via

.. code-block:: bash

    tune --nnodes 1 --nproc_per_node 2 lora_finetune --config alpaca_llama2_lora_finetune\
    --override lora_attn_modules='q_proj,k_proj,v_proj,output_proj'\
    lora_rank=16 output_dir=./lora_experiment_1
