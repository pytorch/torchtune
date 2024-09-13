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

Instruct dataset format
-----------------------

Instruct datasets typically follow the input-output format, where the user prompt is in one column
and the assistant prompt is in another column.

.. code-block:: text

    |  input          |  output          |
    |-----------------|------------------|
    | "user prompt"   | "model response" |

The user input column is converted to a :class:`~torchtune.data.Message` with ``role="user"`` and the assistant
output column is converted to a a :class:`~torchtune.data.Message` with ``role="assistant"``.

.. code-block:: python

    from torchtune.data import Message

    sample = {
        "input": "user prompt",
        "output": "model response",
    }
    #
    # After message transform
    #
    msgs = [
        Message(role="user", content="user prompt"),
        Message(role="assistant", content="model response")
    ]

These are then tokenized by the model tokenizer with the appropriate model-specific special tokens added
(such as beginning-of-sequence, end-of-sequence, and others).

.. code-block:: python

    from torchtune.models.phi3 import phi3_mini_tokenizer

    p_tokenizer = phi3_mini_tokenizer("/tmp/Phi-3-mini-4k-instruct/tokenizer.model")
    tokens, mask = p_tokenizer.tokenize_messages(msgs)
    print(tokens)
    # [1, 32010, 29871, 13, 1792, 9508, 32007, 29871, 13, 32001, 29871, 13, 4299, 2933, 32007, 29871, 13]
    print(p_tokenizer.decode(tokens))
    # '\nuser prompt \n \nmodel response \n'

As an example, you can see the schema of the `C4 200M dataset <https://huggingface.co/datasets/liweili/c4_200m>`_.

Loading instruct datasets from Hugging Face
-------------------------------------------

You simply need to pass in the dataset repo name to ``source``, which is then passed into Hugging Face's ``load_dataset``.
For most datasets, you will also need to specify the ``split``.

.. code-block:: python

    from torchtune.models.gemma import gemma_tokenizer
    from torchtune.datasets import instruct_dataset

    g_tokenizer = gemma_tokenizer("/tmp/gemma-7b/tokenizer.model")
    ds = instruct_dataset(
        tokenizer=g_tokenizer,
        source="liweili/c4_200m",
        split="train"
    )

.. code-block:: yaml

    # Tokenizer is passed into the dataset in the recipe
    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: liweili/c4_200m
      split: train

This will use the default column names "input" and "output". To change the column names, use the ``column_map`` argument.

Loading local and remote instruct datasets
------------------------------------------

To load in a local or remote dataset via https that follows the instruct format, you need to specify the ``source``, ``data_files`` and ``split``
arguments. See Hugging Face's ``load_dataset`` `documentation <https://huggingface.co/docs/datasets/main/en/loading#local-and-remote-files>`_
for more details on loading local or remote files.

.. code-block:: python

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

    # Tokenizer is passed into the dataset in the recipe
    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: json
      data_files: data/my_data.json
      split: train

.. _column_map:

Renaming columns
----------------

If the default column names are "input", "output" and you need to change them to something else,
such as "prompt", "response" for example, define ``column_map`` as such:

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

.. _train_on_input:

Training on user input
----------------------

By default, user prompts are masked from the loss. To override this behavior, set ``train_on_input=True``.

.. code-block:: python

    from torchtune.models.gemma import gemma_tokenizer
    from torchtune.datasets import instruct_dataset

    g_tokenizer = gemma_tokenizer("/tmp/gemma-7b/tokenizer.model")
    ds = instruct_dataset(
        tokenizer=g_tokenizer,
        source="liweili/c4_200m",
        split="train",
        train_on_input=True,
    )

.. code-block:: yaml

    # Tokenizer is passed into the dataset in the recipe
    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: liweili/c4_200m
      split: train
      train_on_input: True

.. _system_prompt:

Adding system prompts
---------------------

By specifying a system prompt, you will prepend a system :class:`~torchtune.data.Message` to each sample
in your dataset.

.. code-block:: python

    from torchtune.models.gemma import gemma_tokenizer
    from torchtune.datasets import instruct_dataset

    g_tokenizer = gemma_tokenizer("/tmp/gemma-7b/tokenizer.model")
    ds = instruct_dataset(
        tokenizer=g_tokenizer,
        source="liweili/c4_200m",
        split="train",
        new_system_prompt="You are a friendly AI assistant",
    )

.. code-block:: yaml

    # Tokenizer is passed into the dataset in the recipe
    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: liweili/c4_200m
      split: train
      new_system_prompt: You are a friendly AI assistant

.. _instruct_template:

Instruct templates
------------------

Typically for instruct datasets, you will want to add a :class:`~torchtune.data.PromptTemplate` to provide task-relevant
information. For example, for a grammar correction task, we may want to use a prompt template like :class:`~torchtune.data.GrammarErrorCorrectionTemplate`
to structure each of our samples. Prompt templates are passed into the tokenizer and automatically applied to the dataset
you are fine-tuning on.

.. code-block:: text

    Correct this to standard English: {user_message}
    ---
    Corrected: {assistant_message}

.. code-block:: python

    from torchtune.models.gemma import gemma_tokenizer
    from torchtune.datasets import instruct_dataset

    g_tokenizer = gemma_tokenizer(
        path="/tmp/gemma-7b/tokenizer.model",
        prompt_template="torchtune.data.GrammarErrorCorrectionTemplate",
    )
    ds = instruct_dataset(
        tokenizer=g_tokenizer,
        source="liweili/c4_200m",
        split="train",
    )

.. code-block:: yaml

    # Tokenizer is passed into the dataset in the recipe
    tokenizer:
      _component_: torchtune.models.gemma.gemma_tokenizer
      path: /tmp/gemma-7b/tokenizer.model
      prompt_template: torchtune.data.GrammarErrorCorrectionTemplate

    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: liweili/c4_200m
      split: train

Example datasets
----------------
- :class:`~torchtune.datasets.alpaca_dataset`
- :class:`~torchtune.datasets.grammar_dataset`
- :class:`~torchtune.datasets.samsum_dataset`
