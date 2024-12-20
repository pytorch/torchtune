.. _chat_dataset_usage_label:

=============
Chat Datasets
=============

Chat datasets involve multi-turn conversations (multiple back-and-forths) between user and assistant.

.. code-block:: python

    [
        {"role": "user", "content": "What is the answer to the ultimate question of life?"},
        {"role": "assistant", "content": "The answer is 42."},
        {"role": "user", "content": "That's ridiculous"},
        {"role": "assistant", "content": "Oh I know."},
    ]

This is more structured than freeform text association that models are typically pre-trained with,
where they learn to simply predict the next token instead of responding accurately to the user.

The primary entry point for fine-tuning with chat datasets in torchtune is the :func:`~torchtune.datasets.chat_dataset`
builder. This lets you specify a local or Hugging Face dataset that follows the chat data format
directly from the config and train your LLM on it.

.. _example_chat:

Example chat dataset
--------------------

.. code-block:: python

    # data/my_data.json
    [
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "What is the answer to life?"
                },
                {
                    "from": "gpt",
                    "value": "The answer is 42."
                },
                {
                    "from": "human",
                    "value": "That's ridiculous"
                },
                {
                    "from": "gpt",
                    "value": "Oh I know."
                }
            ]
        }
    ]

.. code-block:: python

    from torchtune.models.mistral import mistral_tokenizer
    from torchtune.datasets import chat_dataset

    m_tokenizer = mistral_tokenizer(
        path="/tmp/Mistral-7B-v0.1/tokenizer.model",
        prompt_template="torchtune.models.mistral.MistralChatTemplate",
        max_seq_len=8192,
    )
    ds = chat_dataset(
        tokenizer=m_tokenizer,
        source="json",
        data_files="data/my_data.json",
        split="train",
        conversation_column="conversations",
        conversation_style="sharegpt",
        # By default, user prompt is ignored in loss. Set to True to include it
        train_on_input=True,
        new_system_prompt=None,
    )
    tokenized_dict = ds[0]
    tokens, labels = tokenized_dict["tokens"], tokenized_dict["labels"]
    print(m_tokenizer.decode(tokens))
    # [INST] What is the answer to life?  [/INST] The answer is 42. [INST] That's ridiculous  [/INST] Oh I know.
    print(labels)
    # [1, 733, 16289, 28793, 1824, 349, 272, 4372, ...]

.. code-block:: yaml

    # In config
    tokenizer:
      _component_: torchtune.models.mistral.mistral_tokenizer
      path: /tmp/Mistral-7B-v0.1/tokenizer.model
      prompt_template: torchtune.models.mistral.MistralChatTemplate
      max_seq_len: 8192

    dataset:
      _component_: torchtune.datasets.chat_dataset
      source: json
      data_files: data/my_data.json
      split: train
      conversation_column: conversations
      conversation_style: sharegpt
      train_on_input: True
      new_system_prompt: null

Chat dataset format
-------------------

Chat datasets typically have a single column named "conversations" or "messages" that contains a list of messages on a single topic
per sample. The list of messages could include a system prompt, multiple turns between user and assistant, and tool calls/returns.

.. code-block:: text

    |  conversations                                               |
    |--------------------------------------------------------------|
    | [{"role": "user", "content": "What day is today?"},          |
    |  {"role": "assistant", "content": "It is Tuesday."}]         |
    | [{"role": "user", "content": "What about tomorrow?"},        |
    |  {"role": "assistant", "content": "Tomorrow is Wednesday."}] |

As an example, you can see the schema of the `SlimOrca dataset <https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup>`_.

Loading chat datasets from Hugging Face
---------------------------------------

You need to pass in the dataset repo name to ``source``, select one of the conversation styles in ``conversation_style``, and specify the ``conversation_column``.
For most HF datasets, you will also need to specify the ``split``.

.. code-block:: python

    from torchtune.models.gemma import gemma_tokenizer
    from torchtune.datasets import chat_dataset

    g_tokenizer = gemma_tokenizer("/tmp/gemma-7b/tokenizer.model")
    ds = chat_dataset(
        tokenizer=g_tokenizer,
        source="Open-Orca/SlimOrca-Dedup",
        conversation_column="conversations",
        conversation_style="sharegpt",
        split="train",
    )

.. code-block:: yaml

    # Tokenizer is passed into the dataset in the recipe
    dataset:
      _component_: torchtune.datasets.chat_dataset
      source: Open-Orca/SlimOrca-Dedup
      conversation_column: conversations
      conversation_style: sharegpt
      split: train


Loading local and remote chat datasets
--------------------------------------

To load in a local or remote dataset via https that has conversational data, you need to additionally specify the ``data_files`` and ``split``
arguments. See Hugging Face's ``load_dataset`` `documentation <https://huggingface.co/docs/datasets/main/en/loading#local-and-remote-files>`_
for more details on loading local or remote files.

.. code-block:: python

    from torchtune.models.gemma import gemma_tokenizer
    from torchtune.datasets import chat_dataset

    g_tokenizer = gemma_tokenizer("/tmp/gemma-7b/tokenizer.model")
    ds = chat_dataset(
        tokenizer=g_tokenizer,
        source="json",
        conversation_column="conversations",
        conversation_style="sharegpt",
        data_files="data/my_data.json",
        split="train",
    )

.. code-block:: yaml

    # Tokenizer is passed into the dataset in the recipe
    dataset:
      _component_: torchtune.datasets.chat_dataset
      source: json
      conversation_column: conversations
      conversation_style: sharegpt
      data_files: data/my_data.json
      split: train

Specifying conversation style
-----------------------------

The structure of the conversation in the raw dataset can vary widely with different role names and different fields
indicating the message content name. There are a few standardized formats that are common across many datasets.
We have built-in converters to convert these standardized formats into a list of torchtune :class:`~torchtune.data.Message`
that follows this format:

.. code-block:: python

    [
        {
            "role": "system" | "user" | "assistant" | "ipython",
            "content": <message>,
        },
        ...
    ]

.. _sharegpt:

``"sharegpt"``
^^^^^^^^^^^^^^
The associated message transform is :class:`~torchtune.data.ShareGPTToMessages`. The expected format is:

.. code-block:: python

    {
        "conversations": [
            {
                "from": "system" | "human" | "gpt",
                "value": <message>,
            },
            ...
        ]
    }

You can specify ``conversation_style=sharegpt`` in code or config:

.. code-block:: python

    from torchtune.models.gemma import gemma_tokenizer
    from torchtune.datasets import chat_dataset

    g_tokenizer = gemma_tokenizer("/tmp/gemma-7b/tokenizer.model")
    ds = chat_dataset(
        tokenizer=g_tokenizer,
        source="json",
        conversation_column="conversations",
        conversation_style="sharegpt",
        data_files="data/my_data.json",
        split="train",
    )

.. code-block:: yaml

    # Tokenizer is passed into the dataset in the recipe
    dataset:
      _component_: torchtune.datasets.chat_dataset
      source: json
      conversation_column: conversations
      conversation_style: sharegpt
      data_files: data/my_data.json
      split: train

``"openai"``
^^^^^^^^^^^^
The associated message transform is :class:`~torchtune.data.OpenAIToMessages`. The expected format is:

.. code-block:: python

    {
        "messages": [
            {
                "role": "system" | "user" | "assistant",
                "content": <message>,
            },
            ...
        ]
    }

You can specify ``conversation_style=openai`` in code or config:

.. code-block:: python

    from torchtune.models.gemma import gemma_tokenizer
    from torchtune.datasets import chat_dataset

    g_tokenizer = gemma_tokenizer("/tmp/gemma-7b/tokenizer.model")
    ds = chat_dataset(
        tokenizer=g_tokenizer,
        source="json",
        conversation_column="conversations",
        conversation_style="openai",
        data_files="data/my_data.json",
        split="train",
    )

.. code-block:: yaml

    # Tokenizer is passed into the dataset in the recipe
    dataset:
      _component_: torchtune.datasets.chat_dataset
      source: json
      conversation_column: conversations
      conversation_style: openai
      data_files: data/my_data.json
      split: train

If your dataset does not fit one of the above conversation styles, then you will need to create a custom message transform.


Renaming columns
----------------

To specify the column that contains your conversation data, use ``conversation_column``.

.. code-block:: python

    # data/my_data.json
    [
        {
            "dialogue": [
                {
                    "from": "human",
                    "value": "What is the answer to life?"
                },
                {
                    "from": "gpt",
                    "value": "The answer is 42."
                },
                {
                    "from": "human",
                    "value": "That's ridiculous"
                },
                {
                    "from": "gpt",
                    "value": "Oh I know."
                }
            ]
        }
    ]

.. code-block:: python

    from torchtune.models.gemma import gemma_tokenizer
    from torchtune.datasets import chat_dataset

    g_tokenizer = gemma_tokenizer("/tmp/gemma-7b/tokenizer.model")
    ds = chat_dataset(
        tokenizer=g_tokenizer,
        source="json",
        conversation_column="dialogue",
        conversation_style="sharegpt",
        data_files="data/my_data.json",
        split="train",
    )

.. code-block:: yaml

    # Tokenizer is passed into the dataset in the recipe
    dataset:
      _component_: torchtune.datasets.chat_dataset
      source: json
      conversation_column: dialogue
      conversation_style: sharegpt
      data_files: data/my_data.json
      split: train


Chat templates
--------------

Chat templates are defined the same way as instruct templates in :func:`~torchtune.datasets.instruct_dataset`. See :ref:`instruct_template` for more info.


Built-in chat datasets
----------------------
- :class:`~torchtune.datasets.slimorca_dataset`
