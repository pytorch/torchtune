.. _packing_usage_label:

==============
Sample packing
==============

Sample packing involves concatenating multiple samples from your dataset into a single sequence, upto a maximum
sequence length. This requires some pre-processing of the dataset which may
slow down time-to-first-batch, but can introduce significant training speedups
depending on the dataset. In torchtune, sample packing is done by iterating through your dataset and performing
greedy packing upon dataset initialization. You can use sample packing with any of the single dataset builders by passing in
:code:`packed=True`.

To set the max sequence length to pack to, make sure to define ``max_seq_len`` on your tokenizer.

.. code-block:: python

    from torchtune.datasets import alpaca_dataset, PackedDataset
    from torchtune.models.llama3 import llama3_tokenizer

    # Load in tokenizer
    tokenizer = llama3_tokenizer(
        path="/tmp/Llama-3.2-1B-Instruct/original/tokenizer.model",
        max_seq_len=8192,
    )
    dataset = alpaca_dataset(
        tokenizer=tokenizer,
        packed=True,
    )
    print(isinstance(dataset, PackedDataset))  # True

.. code-block:: yaml

    # YAML config
    tokenizer:
      _component_: torchtune.models.llama3.llama3_tokenizer
      path: /tmp/Llama-3.2-1B-Instruct/original/tokenizer.model
      max_seq_len: 8192

    dataset:
      _component_: torchtune.datasets.alpaca_dataset
      packed: True

.. code-block:: bash

    # Command line
    tune run full_finetune_single_device --config llama3_2/1B_full_single_device \
    dataset.packed=True tokenizer.max_seq_len=8192

torchtune will automatically handle document masking and relative position IDs when sample packing is enabled
to prevent different irrelevant samples from cross-attending. This is done via PyTorch's `Flex Attention <https://pytorch.org/blog/flexattention/#document-maskingjagged-sequences>`_,
which enables the use of flash attention with non-causal masks. If your hardware does not support Flex Attention
(for CUDA devices, it must be Turing or above), standard SDPA with memory-efficient attention will be used as a fallback,
while retaining the document masking and relative position IDs.
