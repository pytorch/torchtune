.. _dataset_tutorial_label:

====================================
Configuring Datasets for Fine-Tuning
====================================

This tutorial will guide you through how to set up a dataset to fine-tune on.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * How to quickly get started with built-in datasets
      * How to configure existing dataset classes from the config
      * How to fully customize your own dataset

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Know how to :ref:`configure components from the config<config_tutorial_label>`

Datasets are a core component of fine-tuning workflows that serve as a "steering
wheel" to guide LLM generation for a particular use case. Many publicly shared
open-source datasets have become popular for fine-tuning LLMs and serve as a great
starting point to train your model. We support several widely used datasets to help
quickly bootstrap your fine-tuning. Let's walk through how to set up a common one for fine-tuning.

You can easily specify a dataset directly from the config file:

.. code-block:: yaml

  # Dataset
  dataset:
    _component_: torchtune.datasets.alpaca_dataset

This will indicate to the recipes to create a dataset object that iterates over samples
from `tatsu-lab/alpaca on HuggingFace datasets <https://huggingface.co/datasets/tatsu-lab/alpaca>`_.

We also expose common knobs to tweak the dataset for your needs. For example, let's say
you'd like to reduce the memory footprint of each batch without changing the batch size.
You could tweak :code:`max_seq_len` to achieve that directly from the config.

.. code-block:: yaml

  # Dataset
  dataset:
    _component_: torchtune.datasets.alpaca_dataset
    # Original is 512
    max_seq_len: 256

It is also possible to train on multiple datasets by combining them into a single :class:`~torchtune.datasets.ConcatDataset`. For example:

.. code-block:: yaml

  dataset:
    - _component_: torchtune.datasets.instruct_dataset
      source: vicgalle/alpaca-gpt4
      template: AlpacaInstructTemplate
      split: train
      train_on_input: True
    - _component_: torchtune.datasets.instruct_dataset
      source: samsum
      template: SummarizeTemplate
      column_map: {"output": "summary"}
      split: train
      train_on_input: False

The preceding snippet demonstrates how you can configure each individual dataset's parameters, then combine them into a single concatenated dataset for training.

Customizing instruct templates
------------------------------

To fine-tune an LLM on a particular task, a common approach is to create a fixed instruct
template that guides the model to generate output with a specific goal. Instruct templates
are simply flavor text that structures your inputs for the model. It is model agnostic
and is tokenized normally just like any other text, but it can help condition the model
to respond better to an expected format. For example, the :class:`~torchtune.data.AlpacaInstructTemplate`
structures the data in the following way:

.. code-block:: python

    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

Here is an example of sample that is formatted with :class:`~torchtune.data.AlpacaInstructTemplate`:

.. code-block:: python

    from torchtune.data import AlpacaInstructTemplate

    sample = {
        "instruction": "Classify the following into animals, plants, and minerals",
        "input": "Oak tree, copper ore, elephant",
    }
    prompt = AlpacaInstructTemplate.format(sample)
    print(prompt)
    # Below is an instruction that describes a task, paired with an input that provides further context.
    # Write a response that appropriately completes the request.
    #
    # ### Instruction:
    # Classify the following into animals, plants, and minerals
    #
    # ### Input:
    # Oak tree, copper ore, elephant
    #
    # ### Response:
    #

We provide `other instruct templates <https://github.com/pytorch/torchtune/blob/main/torchtune/data/_instruct_templates.py>`_
for common tasks such summarization and grammar correction. If you need to create your own
instruct template for a custom task, you can create your own :class:`~torchtune.data.InstructTemplate`
class and point to it in the config.

.. code-block:: yaml

    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: mydataset/onthehub
      template: CustomTemplate
      train_on_input: True
      max_seq_len: 512

Customizing chat formats
------------------------
Chat formats are similar to instruct templates, except that they format system,
user, and assistant messages in a list of messages (see :class:`~torchtune.data.ChatFormat`)
for a conversational dataset. These can be configured quite similarly to instruct
datasets.

.. code-block:: yaml

    dataset:
      _component_: torchtune.datasets.chat_dataset
      source: Open-Orca/SlimOrca-Dedup
      conversation_style: sharegpt
      chat_format: Llama2ChatFormat

Here is how messages would be formatted using the :class:`~torchtune.data.Llama2ChatFormat`:

.. code-block:: python

    from torchtune.data import Llama2ChatFormat, Message

    messages = [
        Message(
            role="system",
            content="You are a helpful, respectful, and honest assistant.",
        ),
        Message(
            role="user",
            content="I am going to Paris, what should I see?",
        ),
        Message(
            role="assistant",
            content="Paris, the capital of France, is known for its stunning architecture..."
        ),
    ]
    formatted_messages = Llama2ChatFormat.format(messages)
    print(formatted_messages)
    # [
    #     Message(
    #         role="user",
    #         content="[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n"
    #         "I am going to Paris, what should I see? [/INST] ",
    #     ),
    #     Message(
    #         role="assistant",
    #         content="Paris, the capital of France, is known for its stunning architecture..."
    #     ),
    # ]

Note that the system message is now incorporated in the user message. If you create custom ChatFormats
you can also add more advanced behavior.

Fully customized datasets
-------------------------

More advanced tasks and dataset formats may require you to create your own dataset
class for more flexibility. Let's walk through the :class:`~torchtune.datasets.PreferenceDataset`,
which has custom functionality for RLHF preference data, to understand what you'll need to do.

If you take a look at the code for the :class:`~torchtune.datasets.PreferenceDataset` class,
you'll notice it's quite similar to :class:`~torchtune.datasets.InstructDataset` with a few
adjustments for chosen and rejected samples in preference data.

.. code-block:: python

    chosen_message = [
        Message(role="user", content=prompt, masked=True),
        Message(role="assistant", content=transformed_sample[key_chosen]),
    ]
    rejected_message = [
        Message(role="user", content=prompt, masked=True),
        Message(role="assistant", content=transformed_sample[key_rejected]),
    ]

    chosen_input_ids, c_masks = self._tokenizer.tokenize_messages(
        chosen_message, self.max_seq_len
    )
    chosen_labels = list(
        np.where(c_masks, CROSS_ENTROPY_IGNORE_IDX, chosen_input_ids)
    )

    rejected_input_ids, r_masks = self._tokenizer.tokenize_messages(
        rejected_message, self.max_seq_len
    )
    rejected_labels = list(
        np.where(r_masks, CROSS_ENTROPY_IGNORE_IDX, rejected_input_ids)
    )

If any of the existing dataset classes do not serve your purposes, you can similarly
use one of them as a starting point and add the functionality you need.

To be able to use your custom dataset from the config, you will need to create
a builder function. This is the builder function for the :func:`~torchtune.datasets.stack_exchanged_paired_dataset`,
which creates a :class:`~torchtune.datasets.PreferenceDataset` configured to use
a paired dataset from Hugging Face. Notice that we've also had
to add a custom instruct template as well.

.. code-block:: python

    def stack_exchanged_paired_dataset(
        tokenizer: Tokenizer,
        max_seq_len: int = 1024,
    ) -> PreferenceDataset:
        return PreferenceDataset(
            tokenizer=tokenizer,
            source="lvwerra/stack-exchange-paired",
            template=StackExchangedPairedTemplate(),
            column_map={
                "prompt": "question",
                "chosen": "response_j",
                "rejected": "response_k",
            },
            max_seq_len=max_seq_len,
            split="train",
            data_dir="data/rl",
        )

Now we can easily specify our custom dataset from the config.

.. code-block:: yaml

    # This is how you would configure the Alpaca dataset using the builder
    dataset:
      _component_: torchtune.datasets.stack_exchanged_paired_dataset
      max_seq_len: 512
