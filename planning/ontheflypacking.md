### What:
Packing is the process of putting together samples until a certain target size is reached. This is done to reduce the number of padding tokens in a batch. To avoid contamination between samples, we use a document-level causal mask. To make it faster, we use flex attention to handle the special mask.

Example:
```python
# The current pack with one sample
pack = {"tokens": [1, 2], "labels": [3, 4], "document_ids": [0, 0], "input_pos": [0, 1]}

# The next sample to be added
sample = {"tokens": [5, 6], "labels": [7, 8]}

# After adding the sample
added_docs = add_sample_to_pack(pack, sample, next_doc_id=1)
print(pack)
>>> {"tokens": [1, 2, 5, 6],
    "labels": [3, 4, 7, 8],
    "document_ids": [0, 0, 1, 1],
    "input_pos": [0, 1, 0, 1]}

create_block_causal_mask(document_ids)
>>> [
     [1, 0, 0, 0],
     [1, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 1, 1],
    ]
```

### Goal:
0) Make packing a first-class citizen in TorchTune, available for all sorts of models and recipes.

###  Context:
1) We currently have map-style packing. We pre-process the dataset before training, which is not scalable.
2) Packing is only present for SFT + text data. There is no contract for how to extend it to multimodal, DPO, etc.
3) Collate function has to be aware of packing logic. This is currently hardcoded in the recipe with if/else.

### Solution:
4) Implement a new on-the-fly packing that takes any iterable dataset as input;
5) Packing contract consists of
    i) a `PackingStrategy` that defines a) how to pack and b) the **_mask_mod** used for flex attention;
    ii) a `IterablePackedDataset` that takes any a) `PackingStrategy`, b) **iterable dataset** as input and yields packed samples;
    iii) a `packed_collate_fn` that takes the batch of packed samples and a **mask_fn** (e.g. `strategy.create_block_mask`) to generate the attention mask on the fly.
   To define a new packing strategy, the user only needs to implement the `PackingStrategy` class.

### Implementation:
6) Updated `full_finetune_distributed.py` to use `IterablePackedDataset` when packing is enabled. There are challenges related to iterable datasets and this will be tackled in a separate iterable dataset PR. Changes made were to enable it to run for this RFC.

### Not in this PR:
7) **Logging**: Since we cannot do len(iterable_dataset), we need to add proper logging/metadata to assist users in understanding how far along they are on each dataset and metrics regarding the samples (avg num tokens, avg num samples / pack, etc.)
8) **Packing-aware Loss**: For SFT, the same loss works for map-style and packing. This is not the case for DPO/GRPO, which would need different masking. Future work will have to handle how to associate packing with a loss that supports it.
9) **Packing-aware metrics**: Advanced metrics, such as logprob per sample, would require to be aware of packing;
10) **tokenization**: For advanced packing, e.g. shared prompts in GRPO/DPO, we will need extra metadata from upstream datasets, e.g. prompt len.