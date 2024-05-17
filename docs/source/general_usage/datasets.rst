.. _dataset_usage_label:

========
Datasets
========

torchtune gives you the tools to download external community datasets, load in custom
local datasets, or create your own datasets.

Built-in datasets
-----------------

To use one of the built-in datasets in the library, simply import and call the dataset builder
function. You can see a list of all supported datasets :ref:`here<datasets_label>`.

In code:
.. code-block:: python

    from torchtune.datasets import alpaca_dataset

    # Load in tokenizer
    tokenizer = ...
    dataset = alpaca_dataset(tokenizer)

In config:
.. code-block:: yaml

  # Dataset
  dataset:
    _component_: torchtune.datasets.alpaca_dataset

In CLI:
.. code-block:: bash

    tune run full_finetune_single_device --config llama3/8B_full_single_device dataset=torchtune.datasets.alpaca_dataset

Setting max sequence length
---------------------------

In code:
.. code-block:: python

    from torchtune.datasets import alpaca_dataset

    # Load in tokenizer
    tokenizer = ...
    dataset = alpaca_dataset(
        tokenizer=tokenizer,
        max_seq_len=4096,
    )

In config:
.. code-block:: yaml

  # Dataset
  dataset:
    _component_: torchtune.datasets.alpaca_dataset
    max_seq_len: 4096

In CLI:
.. code-block:: bash

    tune run full_finetune_single_device --config llama3/8B_full_single_device dataset.max_seq_len=4096

Sample packing
--------------

In code:
.. code-block:: python

    from torchtune.datasets import alpaca_dataset, PackedDataset

    # Load in tokenizer
    tokenizer = ...
    dataset = alpaca_dataset(
        tokenizer=tokenizer,
        packed=True,
    )
    print(isinstance(dataset, PackedDataset))  # True

In config:
.. code-block:: yaml

  # Dataset
  dataset:
    _component_: torchtune.datasets.alpaca_dataset
    packed: True

In CLI:
.. code-block:: bash

    tune run full_finetune_single_device --config llama3/8B_full_single_device dataset.packed=True

Custom instruct dataset and instruct templates
----------------------------------------------

If you have a custom instruct dataset that's not already provided in the library,
you can use the :func:`~torchtune.datasets.instruct_dataset` builder and specify
the source path. Instruct datasets typically have multiple columns with text that
are formatted into a prompt template.

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

Here is an example of a sample that is formatted with :class:`~torchtune.data.AlpacaInstructTemplate`:

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
instruct template for a custom task, you can inherit from :class:`~torchtune.data.InstructTemplate`
and create your own class.

In code:
.. code-block:: python

    from torchtune.datasets import instruct_dataset
    from torchtune.data import InstructTemplate

    class CustomTemplate(InstructTemplate):
        # Define the template as string with {} as placeholders for data columns
        template = ...

        # Implement this method
        @classmethod
        def format(
            cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
        ) -> str:
            ...

    # Load in tokenizer
    tokenizer = ...
    dataset = instruct_dataset(
        tokenizer=tokenizer,
        source="my/dataset/path",
        template="CustomTemplate",
    )

In config:
.. code-block:: yaml

  # Dataset
  dataset:
    _component_: torchtune.datasets.instruct_dataset
    source: my/dataset/path
    template: CustomTemplate

In CLI:
.. code-block:: bash

    tune run full_finetune_single_device --config llama3/8B_full_single_device dataset=torchtune.datasets.instruct_dataset dataset.source=my/dataset/path dataset.template=CustomTemplate

.. code-block:: yaml

    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: mydataset/onthehub
      template: CustomTemplate
      train_on_input: True
      max_seq_len: 512


Custom chat dataset and chat formats
------------------------------------

If you have a custom chat/conversational dataset that's not already provided in the library,
you can use the :func:`~torchtune.datasets.chat_dataset` builder and specify
the source path. Chat datasets typically have a single column with multiple back
and forth messages between the user and assistant.

Chat formats are similar to instruct templates, except that they format system,
user, and assistant messages into a list of messages (see :class:`~torchtune.data.ChatFormat`)
for a conversational dataset. These can be configured quite similarly to instruct
datasets.

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

In code:
.. code-block:: python

    from torchtune.datasets import chat_dataset
    from torchtune.data import ChatFormat

    class CustomChatFormat(ChatFormat):
        # Define templates for system, user, assistant messages
        # as strings with {} as placeholders for message content
        system = ...
        user = ...
        assistant = ...

        # Implement this method
        @classmethod
        def format(
            cls,
            sample: List[Message],
        ) -> List[Message]:
            ...

    # Load in tokenizer
    tokenizer = ...
    dataset = chat_dataset(
        tokenizer=tokenizer,
        source="my/dataset/path",
        conversation_style="openai",
        chat_format="CustomChatFormat",
    )

In config:
.. code-block:: yaml

  # Dataset
  dataset:
    _component_: torchtune.datasets.chat_dataset
    source: my/dataset/path
    conversation_style: openai
    chat_format: CustomChatFormat

In CLI:
.. code-block:: bash

    tune run full_finetune_single_device --config llama3/8B_full_single_device dataset=torchtune.datasets.chat_dataset dataset.source=my/dataset/path dataset.conversation_style=openai dataset.chat_format=CustomChatFormat


Multiple in-memory datasets
---------------------------

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

Local datasets
--------------

To use a dataset saved on your local hard drive, simply specify the file type for
``source`` and pass in the ``data_files`` argument using any of the dataset
builder functions. We support

In code:
.. code-block:: python

    from torchtune.datasets import instruct_dataset

    # Load in tokenizer
    tokenizer = ...
    dataset = instruct_dataset(
        tokenizer=tokenizer,
        source="csv",
        template="CustomTemplate"
        data_files="path/to/my/data.csv",
    )

In config:
.. code-block:: yaml

  # Dataset
  dataset:
    _component_: torchtune.datasets.chat_dataset
    source: my/dataset/path
    conversation_style: openai
    chat_format: CustomChatFormat

In CLI:
.. code-block:: bash

    tune run full_finetune_single_device --config llama3/8B_full_single_device dataset=torchtune.datasets.chat_dataset dataset.source=my/dataset/path dataset.conversation_style=openai dataset.chat_format=CustomChatFormat


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
