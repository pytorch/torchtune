.. _sample_packing_deepdive:

==========================================================================================
Fine-tuning in torchtune with sample packing using in-sample position encoding and masking
==========================================================================================

torchtune now supports sample packing for significant boosts in training throughput. This
can be enabled on CLI, in the yaml config, or in any dataset builder by setting ``packed=True``.

.. code-block:: python

    from torchtune.datasets import alpaca_dataset, PackedDataset

    # Load in tokenizer
    tokenizer = ...
    dataset = alpaca_dataset(
        tokenizer=tokenizer,
        packed=True,
    )
    print(isinstance(dataset, PackedDataset))  # True

.. code-block:: yaml

    # YAML config
    dataset:
      _component_: torchtune.datasets.alpaca_dataset
      packed: True

.. code-block:: bash

    # Command line
    tune run full_finetune_single_device --config llama3/8B_full_single_device \
    dataset.packed=True

This will automatically adjust attention masks to avoid cross-sample attending and
update position IDs to be relative, within-sample. In this write-up, we explore what
the expected performance gains you should see are and how in-sample masking and position
encoding can affect fine-tuned model quality.

.. grid:: 1

    .. grid-item-card:: :octicon:`mortar-board;1em;` Highlights

      * Sample packing concatenates multiple data samples into a single packed
      sample up to the maximum sequence length to boost model training throughput.
      * Enabling sample packing for the Stanford Alpaca dataset yielded up to a
      5x boost in throughput (~350 tokens/sec/gpu -> ~1800 tokens/sec/gpu)
      * Adding in-sample position encoding and masking to eliminate cross-attending
      of unrelated samples improved fine-tuned models on hellaswag (+2%) compared
      to packing without in-sample masking, but showed no statistically significant
      difference on truthfulqa_mc2.


Background
----------
Context windows for modern LLMs are ever-increasing, from Llama3 at 8k to
Gemini 1.5 Pro which is pushing 2 million. In practice, many OSS text datasets
(such as Stanford Alpaca) consist of mostly short sequence length samples,
averaging at ~100-1000, which is a small fraction of the full context. This can
lead to significant padding compute and memory overhead, even if ignored or masked
out in the loss function. Sample packing minimizes this overhead by jamming as
many samples as possible in the extra space in the context window. This can massively
increase throughput for model training, since the model will see more data per step.
However, there are several caveats to consider when enabling sample packing for a
transformer decoder model.
1. When concatenating multiple samples in a single context window, we need a way
to tell the model to not attend to all other previous samples and only attend to
the sample the current token belongs to. This can be achieved with a block causal
mask to prevent cross-contamination.
2. Similarly, for position encodings we need to encode relative position of a
token within its sample instead of absolute position across all samples in a
single context window.
3. Ideally, the block causal mask and relative position encoding is compatible
with flash attention to maximize speed-ups.

Masking to avoid cross-contamination during fine-tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default, a causal mask is passed into attention calculations for a transformer
decoder so that each token can attend to all previous tokens but not any future
tokens. This is sound for a single sample, where each token requires the context
provided by all the previous. However, in a packed sample which may contain numerous
different samples only separated by EOS tokens, all consecutive samples should not
attend to previous samples. Instead, we can create a block causal mask where each
sample within a pack has its own causal mask. This way the packs are less noisy
to the model as it only has to focus on attending each sample to itself.

[ADD MASK PICTURE]

Block causal masks have been implemented in several publications (ref and ref) as
a way to directly address issues with cross-contamination. Some users have observed
reasonable performance even in the absence of proper masking and position encoding,
but this lacks systematic investigations. torchtune’s abstraction-free design makes
it possible to add these capabilities and conduct a comparative investigation.

[ADD CODE SNIPPET]

Encoding position within samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Packed samples contain multiple sequences, and thus the typical approach of encoding
position based on absolute index of a token no longer makes sense. Tokens in samples
later in the pack will have the incorrect positions. We need to ensure we encode
relative position of a token within its sample instead of absolute.

[ADD PICTURE DEMONSTRATING RELATIVE POSITIONS AND CODE SNIPPET]

Approach
--------
Setting ``packed=True`` returns a :class:`~torchtune.datasets.PackedDataset ` class
that acts as a token buffer for any arbitrary dataset and creates the block causal
masks and in-sample position ids for each pack. The buffer is created as an offline
pre-processing step prior to training in the follow manner:
- Iterate through the dataset and greedily pack until we hit max sequence length
- Append as a new pack
- Repeat until dataset is exhausted or we hit max_packs specified by the user

.. code-block:: python

    class PackedDataset(Dataset):
    ...
    def _pack(self):
        # buffer to hold samples until they are long enough to be added to self.packs
        current_pack = {
            "tokens": [],
            "labels": [],
            "mask": [],
            "input_pos": [],
        }

        # Iterate over entire dataset
        for sample in self.dataset:
            tokens, labels = sample["tokens"], sample["labels"]
            seq_len = len(tokens)

            # Create integer mask and position ids for current sample and extend
            # current pack
            current_sample = {
                "tokens": tokens,
                "labels": labels,
                # Mask is simply a causal mask within this sample length
                "mask": [torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))],
                # Input pos is the position ids for each token
                "input_pos": list(range(seq_len)),
            }
            current_pack = {k: v + current_sample[k] for k, v in current_pack.items()}

            # If the current pack is long enough, add it to self.packs
            if len(current_pack["tokens"]) > self.max_seq_len:
                current_pack = self._add_pack(
                    current_pack=current_pack,
                    ...
                )

        # Add the last pack with remaining samples that did not fit in previous
        if len(current_pack["tokens"]) > 0:
            current_pack = self._add_pack(
                current_pack=current_pack, ...)
            )

    # In training recipe
    for batch in dataloader:
        logits = model(batch["tokens"], mask=batch["mask"], input_pos=batch["input_pos"])

The major tradeoff here is deciding where to create the cross-contamination
mask and the relative position ids.

1. **Create it sample-wise when packing**. This increases memory usage due to storing
additional masks and ids, but is less intrusive in the model code and faster than
looping through the entire batch in the model forward to determine the masks and ids.
2. **Create it batch-wise on-the-fly**. If we maintain the sequence lengths in every
pack, we can iterate over the batch during forward pass and create the mask and ids
ad-hoc. This is more memory efficient because we don’t have to store the masks and
ids for the entire dataset, but it is slower (iterate over batch dim and sequence
dim to create the masks and ids) and more intrusive code.

We opted for approach (1), because for iterable datasets, we will no longer need
to store the in-sample masks and position ids for the whole dataset in-memory and
create them on-the-fly. Apart from passing the block causal attention mask and the
modified position ids in the model forward, there are no changes that are needed
in the actual model to support sample packing.

Experiments
-----------
To measure the impact of our implementation on performance and model quality, we
tested five fine-tuning configurations.

[INSERT TABLE WITH LAUNCH COMMANDS]

We used the following fine-tuning hyperparameters:
- Dataset: tatsu-lab/alpaca
- Max sequence length: 4096
- Model: llama3 instruct 8B
- Batch size: 4
- Epochs: 3
- LR: 2e-5 - 2e-4
- Distributed: 1x8 A100 80 GB
We varied sequence length to understand how tokens/sec is affected by the context size. All plots were created using WandB.

Training performance
^^^^^^^^^^^^^^^^^^^^
We observed that sample packing significantly reduces single epoch time and improves
performance proportional to max sequence length. At sequence length of 512, tokens/sec
increased three-fold from ~350 to ~1050 per gpu. Increasing the max sequence length
to 4096 further boosted throughput from ~350 tokens/sec to ~1800 tokens/sec, which
is nearly 5x the unpacked version. Similarly, packing at a sequence length of 512
reduced single epoch time from 55 minutes to 21 minutes. At a sequence length of
4096, single epoch time further reduced to 6.5 minutes. The pre-processing time
for packing 52k samples in Alpaca only takes 40 seconds using a single rank. Losses
between packed runs and unpacked runs were identical after accounting for more data
seen in a batch in the packed runs by increasing learning rate.

[INSERT TABLE]

Model quality
^^^^^^^^^^^^^
We ran the EleutherAI eval harness on the fine-tuned models and performed one-tailed
t-tests on the results for truthfulqa_mc2 and hellaswag. Based on the statistics,
we can make the following conclusions:
- Packing with & without in-sample masking and position encoding and not packing perform similarly on truthfulqa_mc2
- Packing with in-sample masking performs better on hellaswag (~2%) compared to packing without in-sample masking
- Packing and not packing have similar performance on both truthfulqa_mc2 and hellaswag

[INSERT TABLE]

Caveats
^^^^^^^
Important caveats to call out:
- Using iterable datasets may require packing on the fly, which means throughput
for packed compared to unpacked may decrease. However, this might be counterbalanced
by packing the dataset offline or with the performance gains by enabling flash
attention in the future
- Each of the configurations were only run once - a more robust analysis would
have multiple runs per configuration with different seeds and aggregate the eval
metrics
- This was only run on the Alpaca dataset which leans towards shorter sample lengths.
Other datasets may yield different performance gains and eval results.

Future directions
-----------------
There are further optimizations we will explore to boost the performance gains
from sample packing and extend it to more datasets.
- Enabling flash attention 2 with in-sample masking
- Shard the packing across devices, which can decrease the pre-processing time
many-fold especially for large datasets
- Compare performance gains between a more optimized bin-packing algorithm for
packing and the naive greedy approach
- Support sample packing on-the-fly for iterable datasets
- Support sample packing for multimodal and interleaved datasets
