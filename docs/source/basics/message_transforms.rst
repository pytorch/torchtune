.. _message_transform_usage_label:

==================
Message Transforms
==================

Message transforms perform the conversion of raw sample dictionaries from your dataset into torchtune's
:class:`~torchtune.data.Message` structure. This is where you, as the architect of your data and training pipeline,
get to place all your data preprocessing and formatting logic without having to worry about the rest of
the pipeline to tokenization and model inputs. We provide utilities to make this as easy as possible,
and if your dataset folows a standard format, you can use our built-in utilities to quickly get
fine-tuning without having to write any data transforms.


Configuring message transforms
------------------------------
Most of our built-in message transforms contain parameters for controlling input masking (``train_on_input``),
adding a system prompt (``new_system_prompt``), and changing the expected column names (``column_map``).
These are exposed in our dataset builders :func:`~torchtune.datasets.instruct_dataset` and :func:`~torchtune.datasets.chat_dataset`
so you don't have to worry about the message transform itself and can configure this directly from the config.
You can view the API ref for these builders for more details.

For example, :func:`~torchtune.datasets.instruct_dataset` uses :class:`~torchtune.data.InputOutputToMessages` as the message transform.
The following code will configure :class:`~torchtune.data.InputOutputToMessages` to train on the user prompt, prepend a custom system
prompt, and use the column names "prompt" and "response" instead.

.. code-block:: python

    # In code
    from torchtune.datasets import instruct_dataset

    ds = instruct_dataset(
        source="json",
        data_files="data/my_data.json",
        split="train",
        train_on_input=True,
        new_system_prompt="You are a friendly AI assistant.",
        column_map={"input": "prompt", "output": "response"},
    )

.. code-block:: yaml

    # In config
    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: json
      data_files: data/my_data.json
      split: train
      train_on_input: True
      new_system_prompt: You are a friendly AI assistant.
      column_map:
        input: prompt
        output: response

Custom message transforms
-------------------------
If our built-in message transforms do not configure for your particular dataset well,
you can create your own class with full flexibility. Simply inherit from the :class:`~torchtune.modules.transforms.Transform`
class and add your code in the ``__call__`` method.

A simple contrived example would be to take one column from the dataset as the user message and another
column as the model response. Indeed, this is quite similar to :class:`~torchtune.data.InputOutputToMessages`.

.. code-block:: python

    from torchtune.modules.transforms import Transform
    from torchtune.data import Message
    from typing import Any, Mapping

    class MessageTransform(Transform):
        def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
            return [
                Message(
                    role="user",
                    content=sample["input"],
                    masked=True,
                    eot=True,
                ),
                Message(
                    role="assistant",
                    content=sample["output"],
                    masked=False,
                    eot=True,
                ),
            ]

    sample = {"input": "hello world", "output": "bye world"}
    transform = MessageTransform()
    messages = transform(sample)
    print(messages)
    # [<torchtune.data._messages.Message at 0x7fb0a10094e0>,
    # <torchtune.data._messages.Message at 0x7fb0a100a290>]
    for msg in messages:
        print(msg.role, msg.text_content)
    # user hello world
    # assistant bye world

See :ref:`creating_messages` for more details on how to manipulate :class:`~torchtune.data.Message` objects.

To use this for your dataset, you must create a custom dataset builder that uses the underlying
dataset class, :class:`~torchtune.datasets.SFTDataset`.

.. code-block:: python

    # In data/dataset.py
    from torchtune.datasets import SFTDataset

    def custom_dataset(tokenizer, **load_dataset_kwargs) -> SFTDataset:
        message_transform = MyMessageTransform()
        return SFTDataset(
            source="json",
            data_files="data/my_data.json",
            split="train",
            message_transform=message_transform,
            model_transform=tokenizer,
            **load_dataset_kwargs,
        )

This can be used directly from the config.

.. code-block:: yaml

    dataset:
      _component_: data.dataset.custom_dataset


Example message transforms
--------------------------
- Instruct
    - :class:`~torchtune.data.InputOutputToMessages`
- Chat
    - :class:`~torchtune.data.ShareGPTToMessages`
    - :class:`~torchtune.data.JSONToMessages`
- Preference
    - :class:`~torchtune.data.ChosenRejectedToMessages`
