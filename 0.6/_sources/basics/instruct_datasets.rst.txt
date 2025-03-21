.. _instruct_dataset_usage_label:

=================
Instruct Datasets
=================

Instruction tuning involves training an LLM to perform specific task(s). This typically takes the form
of a user command or prompt and the assistant's response, along with an optional system prompt that
describes the task at hand. This is more structured than freeform text association that models are
typically pre-trained with, where they learn to specifically predict the next token instead of completing
the task.

The primary entry point for fine-tuning with instruct datasets in torchtune is the :func:`~torchtune.datasets.instruct_dataset`
builder. This lets you specify a local or Hugging Face dataset that follows the instruct data format
directly from the config and train your LLM on it.

.. _example_instruct:

Example instruct dataset
------------------------

Here is an example of an instruct dataset to fine-tune for a grammar correction task.

.. code-block:: bash

    head data/my_data.csv
    # incorrect,correct
    # This are a cat,This is a cat.

.. code-block:: python

    from torchtune.models.gemma import gemma_tokenizer
    from torchtune.datasets import instruct_dataset

    g_tokenizer = gemma_tokenizer(
        path="/tmp/gemma-7b/tokenizer.model",
        prompt_template="torchtune.data.GrammarErrorCorrectionTemplate",
        max_seq_len=8192,
    )
    ds = instruct_dataset(
        tokenizer=g_tokenizer,
        source="csv",
        data_files="data/my_data.csv",
        split="train",
        # By default, user prompt is ignored in loss. Set to True to include it
        train_on_input=True,
        # Prepend a system message to every sample
        new_system_prompt="You are an AI assistant. ",
        # Use columns in our dataset instead of default
        column_map={"input": "incorrect", "output": "correct"},
    )
    tokenized_dict = ds[0]
    tokens, labels = tokenized_dict["tokens"], tokenized_dict["labels"]
    print(g_tokenizer.decode(tokens))
    # You are an AI assistant. Correct this to standard English:This are a cat---\nCorrected:This is a cat.
    print(labels)  # System message is masked out, but not user message
    # [-100, -100, -100, -100, -100, -100, 27957, 736, 577, ...]

.. code-block:: yaml

    # In config
    tokenizer:
      _component_: torchtune.models.gemma.gemma_tokenizer
      path: /tmp/gemma-7b/tokenizer.model
      prompt_template: torchtune.data.GrammarErrorCorrectionTemplate
      max_seq_len: 8192

    dataset:
      source: csv
      data_files: data/my_data.csv
      split: train
      train_on_input: True
      new_system_prompt: You are an AI assistant.
      column_map:
        input: incorrect
        output: correct

Instruct dataset format
-----------------------

Instruct datasets are expected to follow an input-output format, where the user prompt is in one column
and the assistant prompt is in another column.

.. code-block:: text

    |  input          |  output          |
    |-----------------|------------------|
    | "user prompt"   | "model response" |

As an example, you can see the schema of the `C4 200M dataset <https://huggingface.co/datasets/liweili/c4_200m>`_.


Loading instruct datasets from Hugging Face
-------------------------------------------

You simply need to pass in the dataset repo name to ``source``, which is then passed into Hugging Face's ``load_dataset``.
For most datasets, you will also need to specify the ``split``.

.. code-block:: python

    # In code
    from torchtune.models.gemma import gemma_tokenizer
    from torchtune.datasets import instruct_dataset

    g_tokenizer = gemma_tokenizer("/tmp/gemma-7b/tokenizer.model")
    ds = instruct_dataset(
        tokenizer=g_tokenizer,
        source="liweili/c4_200m",
        split="train"
    )

.. code-block:: yaml

    # In config
    tokenizer:
      _component_: torchtune.models.gemma.gemma_tokenizer
      path: /tmp/gemma-7b/tokenizer.model

    # Tokenizer is passed into the dataset in the recipe
    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: liweili/c4_200m
      split: train

This will use the default column names "input" and "output". To change the column names, use the ``column_map`` argument (see :ref:`column_map`).

Loading local and remote instruct datasets
------------------------------------------

To load in a local or remote dataset via https that follows the instruct format, you need to specify the ``source``, ``data_files`` and ``split``
arguments. See Hugging Face's ``load_dataset`` `documentation <https://huggingface.co/docs/datasets/main/en/loading#local-and-remote-files>`_
for more details on loading local or remote files.

.. code-block:: python

    # In code
    from torchtune.models.gemma import gemma_tokenizer
    from torchtune.datasets import instruct_dataset

    g_tokenizer = gemma_tokenizer("/tmp/gemma-7b/tokenizer.model")
    ds = instruct_dataset(
        tokenizer=g_tokenizer,
        source="json",
        data_files="data/my_data.json",
        split="train",
    )

.. code-block:: yaml

    # In config
    tokenizer:
      _component_: torchtune.models.gemma.gemma_tokenizer
      path: /tmp/gemma-7b/tokenizer.model

    # Tokenizer is passed into the dataset in the recipe
    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: json
      data_files: data/my_data.json
      split: train

.. _column_map:

Renaming columns
----------------

You can remap the default column names to the column names in your dataset by specifying
``column_map`` as ``{"<default column>": "<column in your dataset>"}``. The default column names
are detailed in each of the dataset builders (see :func:`~torchtune.datasets.instruct_dataset` and
:func:`~torchtune.datasets.chat_dataset` as examples).

For example, if the default column names are "input", "output" and you need to change them to something else,
such as "prompt", "response", then ``column_map = {"input": "prompt", "output": "response"}``.

.. code-block:: python

    # data/my_data.json
    [
        {"prompt": "hello world", "response": "bye world"},
        {"prompt": "are you a robot", "response": "no, I am an AI assistant"},
        ...
    ]

.. code-block:: python

    from torchtune.models.gemma import gemma_tokenizer
    from torchtune.datasets import instruct_dataset

    g_tokenizer = gemma_tokenizer("/tmp/gemma-7b/tokenizer.model")
    ds = instruct_dataset(
        tokenizer=g_tokenizer,
        source="json",
        data_files="data/my_data.json",
        split="train",
        column_map={"input": "prompt", "output": "response"},
    )

.. code-block:: yaml

    # Tokenizer is passed into the dataset in the recipe
    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: json
      data_files: data/my_data.json
      split: train
      column_map:
        input: prompt
        output: response

.. _instruct_template:

Instruct templates
------------------

Typically for instruct datasets, you will want to add a :class:`~torchtune.data.PromptTemplate` to provide task-relevant
information. For example, for a grammar correction task, we may want to use a prompt template like :class:`~torchtune.data.GrammarErrorCorrectionTemplate`
to structure each of our samples. Prompt templates are passed into the tokenizer and automatically applied to the dataset
you are fine-tuning on. See :ref:`using_prompt_templates` for more details.


Built-in instruct datasets
--------------------------
- :class:`~torchtune.datasets.alpaca_dataset`
- :class:`~torchtune.datasets.grammar_dataset`
- :class:`~torchtune.datasets.samsum_dataset`
