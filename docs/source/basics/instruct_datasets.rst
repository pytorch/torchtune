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
output column is converted to a a :class:`~torchtune.data.Message` with ``role="assistant"``. These are then pass into
the model tokenizer's ``tokenize_messages`` to encode and add appropriate model-specific special tokens.

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
        tokenizer=tokenizer,
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
        tokenizer=tokenizer,
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

Custom instruct datasets
------------------------

Overview of InputOutputToMessages and how to customize it

Instruct templates
------------------

Overview of instruct prompt templates and how to enable them.
