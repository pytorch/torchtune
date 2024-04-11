.. _qlora_finetune_label:

=============================
Finetuning Llama2 with QLoRA
=============================

In this tutorial, we'll learn about `QLoRA <https://arxiv.org/abs/2305.14314>`_, an enhancement on top of
`LoRA <https://arxiv.org/abs/2106.09685>`_ that maintains frozen model parameters in 4-bit quantized precision, thereby reducing memory usage. We'll
walk through how QLoRA can be utilized within TorchTune to finetune a Llama2-7b model in < 10 GB of memory.
It is highly recommended to first develop an understanding of :ref:`LoRA finetuning in TorchTune<lora_finetune_label>`.


.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * How QLoRA saves memory over LoRA finetuning
      * An overview of QLoRA in TorchTune
      * How to run a QLoRA finetune in TorchTune

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Be familiar with :ref:`TorchTune<overview_label>`
      * Make sure to :ref:`install TorchTune<install_label>`
      * Make sure you have downloaded the :ref:`Llama2-7B model weights<download_llama_label>`
      * Be familiar with :ref:`LoRA in torchtune<lora_finetune_label>`

What is QLoRA?
---------------

`QLoRA <https://arxiv.org/abs/2305.14314>`_ builds on top of `LoRA <https://arxiv.org/abs/2106.09685>`_ to enable additional
memory efficiency on top of LoRA. In LoRA, model parameters can be thought of as existing in two partitions: adapters, which are
low-rank matrices added to different layers of a neural network, and base model parameters, which are parameters that are part of
the original model. In vanilla LoRA style training, both these parameters are held in the same precision (typically fp32 or bf16), and
therefore activations and intermediate gradients computed are in fp32/bf16.

QLoRA further quantizes the base model parameters into a bespoke 4-bit NormalFloat (NF4) data type, resulting in 4x less parameter memory usage while
largely retaining model accuracy. As a result, the vast majority of parameters only take up 4 bits (as opposed to 16 or 32 bits by bf16/fp32 dtypes). Adapter
parameters are still held in the original precision, and activations, gradients, and optimizer states still exist in the higher precision to preserve
accuracy.

The `QLoRA paper <https://arxiv.org/abs/2305.14314>`_ introduces two key abstractions to decrease memory usage and avoid accuracy degradation: the bespoke 4-bit NormatFloat
type, and a double quantization method that quantizes the quantization parameters themselves to save even more memory. TorchTune uses
the `NF4Tensor` abstraction from the `TorchAO library <https://github.com/pytorch-labs/ao>`_ to build QLoRA components as specified in the paper.


QLoRA Core Abstractions and Usage
----------------------------------------

In this section, we'll first overview how to apply QLoRA to a `LoRALinear` layer in TorchTune, and then learn about how it works under the hood.

To quantize a `LoRALinear` layer in the QLoRA style, simply pass in the `quantize_base` flag as ``True`` into :class:`~torchtune.modules.peft.LoRALinear`. This flag
will result in base model weights being quantized and backed by the ``NF4Tensor`` dtype. Forward passes will also be automatically handled to work with the ``NF4Tensor`` dtype,
specifically, the ``NF4`` base weight will be de-quantized to bf16, activation will be computed, and only the 4-bit parameter will be stored for gradient computation
in the backward pass, avoiding extra memory usage that would be incurred by storing the full bf16 dtype.

Here's an example of creating a quantized `LoRALinear` layer in comparison to an unquantized `LoRALinear` layer. As we can see, the quantized layer consumes
~8x less memory than the unquantized counterpart.

.. code-block:: python

  import torch
  from torchtune.modules.peft import LoRALinear

  torch.set_default_device("cuda")
  qlora_linear = LoRALinear(512, 512, rank=8, alpha=0.1, quantize_base=True)
  print(torch.cuda.memory_allocated())  # 177,152 bytes
  del qlora_linear
  torch.cuda.empty_cache()
  lora_linear = LoRALinear(512, 512, rank=8, alpha=0.1, quantize_base=False)
  print(torch.cuda.memory_allocated()) # 1,081,344 bytes


Now, we'll peel this back to understand how quantization is done with ``NF4Tensor`` and handled appropriately in the forward pass.

First, we'll begin with
a vanilla minimal LoRA layer, taken from :ref:`the LoRA tutorial <lora_finetune_label>` and augmented to support quantization:

.. code-block:: python

  from torch import nn, Tensor
  import torch.nn.functional as F
  from torchao.dtypes.nf4tensor import linear_nf4, to_nf4

  class LoRALinear(nn.Module):
    def __init__(
      self,
      in_dim: int,
      out_dim: int,
      rank: int,
      alpha: float,
      dropout: float,
      quantize_base: bool
    ):
      # These are the weights from the original pretrained model
      self.linear = nn.Linear(in_dim, out_dim, bias=False)
      self.linear_weight = self.linear.weight
      # Use TorchAO's to_nf4 API to quantize the base weight if needed.
      if quantize_base:
        self.linear_weight = to_nf4(self.linear_weight)
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
      if quantize_base:
        # Call into TorchAO's linear_nf4 to run linear forward pass w/quantized weight.
        frozen_out  = linear_nf4(x, self.weight)
      else:
        frozen_out = F.linear(x, self.weight)

      # lora_a projects inputs down to the much smaller self.rank,
      # then lora_b projects back up to the output dimension
      lora_out = self.lora_b(self.lora_a(self.dropout(x)))

      # Finally, scale by the alpha parameter (normalized by rank)
      # and add to the original model's outputs
      return frozen_out + (self.alpha / self.rank) * lora_out

As mentioned above, TorchTune takes a dependency on `TorchAO library <https://github.com/pytorch-labs/ao>`_ for some of the core components required for QLoRA. This includes the
`NF4Tensor`, as well as helpful utilities including ``to_nf4`` and ``linear_nf4``.

``to_nf4`` accepts an unquantized (bf16 or fp32) tensor and produces an ``NF4`` representation of the weight. See the `implementation <https://github.com/pytorch-labs/ao/blob/c40358072f99b50cd7e58ec11e0e8d90440e3e25/torchao/dtypes/nf4tensor.py#L587>`_ of ``to_nf4`` for more details.
``linear_nf4`` handles the forward pass and autograd when running with quantized base model weights. It computes the forward pass as a regular
``F.linear`` with the incoming activation and unquantized weight. The quantized weight is saved for backward, as opposed to the unquantized version of the weight, to avoid extra
memory usage due to storing higher precision variables to compute gradients in the backward pass. See `linear_nf4 <https://github.com/pytorch-labs/ao/blob/main/torchao/dtypes/nf4tensor.py#L577>`_ for more details.

In the next section, we'll learn about how to use QLoRA in TorchTune to build a QLoRA quantized Llama2-7b model, as well as some nuances around
checkpointing that are important to be aware of to avoid spiking memory usage.


Using QLoRA in TorchTune
----------------------------

TODO this section



Putting it all together: QLoRA finetune
-----------------------------------------

Stuff about how to actually use QLoRA, look at the memory usage etc.
