.. _message_transform_usage_label:

==================
Message Transforms
==================

Message transforms perform the conversion of raw sample dictionaries from your dataset into torchtune's
:class:`~torchtune.data.Message` structure. Once you data is represented as Messages, torchtune will handle
tokenization and preparing it for the model.

.. TODO (rafiayub): place an image here to depict overall pipeline


Configuring message transforms
------------------------------
Most of our built-in message transforms contain parameters for controlling input masking (``masking_strategy``),
adding a system prompt (``new_system_prompt``), and changing the expected column names (``column_map``).
These are exposed in our dataset builders :func:`~torchtune.datasets.instruct_dataset` and :func:`~torchtune.datasets.chat_dataset`
so you don't have to worry about the message transform itself and can configure this directly from the config.
You can see :ref:`example_instruct` or :ref:`example_chat` for more details.

.. _custom_message_transform:

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
    from pprint import pprint

    class MessageTransform(Transform):
        def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
            messages = [
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
            return {"messages": messages}

    input_sample = {"input": "hello world", "output": "bye world"}
    transform = MessageTransform()
    output_sample = transform(input_sample)
    pprint(output_sample)
    # {'messages': [Message(role='user', content=['hello world']),
    #               Message(role='assistant', content=['bye world'])]}

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
    - :class:`~torchtune.data.AlpacaToMessages`
- Chat
    - :class:`~torchtune.data.ShareGPTToMessages`
    - :class:`~torchtune.data.OpenAIToMessages`
- Preference
    - :class:`~torchtune.data.ChosenRejectedToMessages`
