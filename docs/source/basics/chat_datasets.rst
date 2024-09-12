.. _chat_dataset_usage_label:

=============
Chat Datasets
=============

Chat datasets involve multi-turn conversations (multiple back-and-forths) between user and assistant.
This is more structured than freeform text association that models are typically pre-trained with,
where they learn to simply predict the next token instead of responding accurately to the user.

The primary entry point for fine-tuning with chat datasets in torchtune is the :func:`~torchtune.datasets.chat_dataset`
builder. This lets you specify a local or Hugging Face dataset that follows the chat data format
directly from the config and train your LLM on it.

Chat dataset format
-------------------

Chat datasets typically have a single column named "conversations" or "messages" that contains a list of messages on a single topic
per sample. The list of messages could include a system prompt, multiple turns between user and assistant, and tool calls/returns.

.. code-block:: text

    |  conversations                         |
    |----------------------------------------|
    | [{"role": "user", "content": Q1},      |
    |  {"role": "assistant", "content": A1}] |

The conversation structure in the dataset will be reformatted to follow torchtune's :class:`~torchtune.data.Message` structure.
The above will be converted to:

.. code-block:: python

    messages = [
        Message(role="user", content="Q1"),
        Message(role="assistant", content="A1"),
    ]

The list of messages is then passed into the model tokenizer's ``tokenize_messages`` to encode and add appropriate model-specific special tokens.

As an example, you can see the schema of the `SlimOrca dataset <https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup>`_.

Specifying conversation style
-----------------------------

The structure of the conversation in the raw dataset can vary widely with different role names, different fields indicating
where the message content name, and other ways. There are a few standardized formats that are common across many datasets.
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

``"json"``
^^^^^^^^^^
The associated message transform is :class:`~torchtune.data.JSONToMessages`. The expected format is:

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

If your dataset does not fit one of the above conversation styles, then you will need to create a custom message transform.

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
      _component_: torchtune.datasets.instruct_dataset
      source: json
      conversation_column: conversations
      conversation_style: sharegpt
      data_files: data/my_data.json
      split: train

Renaming columns
----------------

You can remap column names similarly to :func:`~torchtune.datasets.instruct_dataset`. See :ref:`column_map` for more info.

Training on user input
----------------------

You can train on user input similarly to :func:`~torchtune.datasets.instruct_dataset`. See :ref:`train_on_input` for more info.

Adding system prompts
---------------------

You can set a system prompt for your dataset similarly to :func:`~torchtune.datasets.instruct_dataset`. See :ref:`system_prompt` for more info.

Chat templates
--------------

Chat templates are defined the same way as instruct templates in :func:`~torchtune.datasets.instruct_dataset`. See :ref:`instruct_template` for more info.
