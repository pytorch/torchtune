.. _dataset_tutorial_label:

======================
Bring Your Own Dataset
======================

This tutorial will guide you through configuring your own dataset to fine-tune on.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * How to configure existing dataset classes from the config
      * How to fully customize your own dataset

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Know how to :ref:`configure components from the config<config_tutorial_label>`

Datasets are a core component of fine-tuning workflows that serve as a "steering
wheel" to guide LLM generation for a particular use case. We provide several common
categories of datasets to help you quickly bootstrap your own data.

- :class:`~torchtune.datasets._instruct.InstructDataset`: primarily used for instruction-following
data, where the input is typically a specific task the user wants the model to do.
It formats the input data into the provided prompt template string (see :class:`~torchtune.data._instruct_templates.InstructTemplate`)
related to the task, so you can quickly customize this to fine-tune on an instruct task.
- :class:`~torchtune.datasets._instruct.ChatDataset`: primarily used for conversational
data with support for multiple turns between user, assistant, and an optional system prompt.
When you configure this, you need to ensure your conversations follow the LLaMA format,
either by formatting the data offline or providing a transform via :code:`convert_to_messages`
(see :func:`~torchtune.data._converters.sharegpt_to_llama2_messages` as an example).

You can configure these classes directly from the config by using the associated
builder functions - :func:`~torchtune.datasets._instruct.instruct_dataset` and :func:`~torchtune.datasets._chat.chat_dataset`.

.. code-block:: yaml

    # This is how you would configure the Alpaca dataset using the builder
    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: tatsu-lab/alpaca
      template: AlpacaInstructTemplate
      train_on_input: True
      max_seq_len: 512

If you need to further customize your dataset, we strongly encourage you to create
your own dataset class and associated builder function so you can configure it
from the yaml config. Then you have full control of how the data is preprocessed
and formatted. You can then use it in the config in a similar way.

.. code-block:: yaml

    # This is how you would configure the Alpaca dataset using the builder
    dataset:
      _component_: torchtune.datasets.my_dataset_builder
      source: mydataset/onthehub
      template: CustomTemplate
      ...

Instruct templates
------------------
For most instruct-based tasks, you'll want to format your prompts into a template
string that describes the task before passing it to the model. The library has
various instruct templates for common tasks that you can either use or take inspiration
from for a custom template (see :class:`~torchtune.data._instruct_templates.InstructTemplate`).

For example, let's say you want to format your dataset into the Alpaca instruction
format. You will need to use :class:`~torchtune.data._instruct_templates.AlpacaInstructTemplate`
which looks like this:

.. code-block:: python

    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

Here is an example of sample that is formatted with :class:`~torchtune.data._instruct_templates.AlpacaInstructTemplate`:

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

Chat formats
------------
Chat formats are similar to instruct templates, except that they format system,
user, and assistant messages in a list of messages (see :class:`~torchtune.data._chat_formats.ChatFormat`).

Here is an example using the :class:`~torchtune.data._chat_formats.Llama2ChatFormat`:

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
