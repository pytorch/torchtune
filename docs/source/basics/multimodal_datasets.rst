.. _multimodal_dataset_usage_label:

===================
Multimodal Datasets
===================

Multimodal datasets include more than one data modality, e.g. text + image, and can be used to train transformer-based models.
torchtune currently only supports multimodal text+image chat datasets for Vision-Language Models (VLMs).

The primary entry point for fine-tuning with multimodal datasets in torchtune is the :func:`~torchtune.datasets.multimodal.multimodal_chat_dataset`
builder. This lets you specify a local or Hugging Face dataset that follows the multimodal chat data format
directly from the config and train your VLM on it.

.. _example_multimodal:

Example multimodal dataset
--------------------------

Here is an example of a multimodal chat dataset for a visual question-answering task. Note that there is a placeholder
in the text, ``"<image>"`` for where to place the image tokens. This will get replaced by the image special token
``<|image|>`` in the example below.

.. code-block:: python

    # data/my_data.json
    [
        {
            "dialogue": [
                {
                    "from": "human",
                    "value": "<image>What time is it on the clock?",
                },
                {
                    "from": "gpt",
                    "value": "It is 10:00 AM.",
                },
            ],
            "image_path": "images/clock.jpg",
        },
        ...,
    ]

.. code-block:: python

    from torchtune.models.llama3_2_vision import llama3_2_vision_transform
    from torchtune.datasets.multimodal import multimodal_chat_dataset

    model_transform = Llama3VisionTransform(
        path="/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model",
        prompt_template="torchtune.data.QuestionAnswerTemplate",
        max_seq_len=8192,
        image_size=560,
    )
    ds = multimodal_chat_dataset(
        model_transform=model_transform,
        source="json",
        data_files="data/my_data.json",
        column_map={
            "dialogue": "conversations",
            "image_path": "image",
        },
        image_dir="/home/user/dataset/",  # /home/user/dataset/images/clock.jpg
        image_tag="<image>",
        split="train",
    )
    tokenized_dict = ds[0]
    print(model_transform.decode(tokenized_dict["tokens"], skip_special_tokens=False))
    # '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nQuestion:<|image|>What time is it on the clock?Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nIt is 10:00AM.<|eot_id|>'
    print(tokenized_dict["encoder_input"]["images"][0].shape)  # (num_tiles, num_channels, tile_height, tile_width)
    # torch.Size([4, 3, 224, 224])

.. code-block:: yaml

    # In config - model_transforms takes the place of the tokenizer
    model_transform:
      _component_: torchtune.models.llama3_2_vision_transform
      path: /tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model
      prompt_template: torchtune.data.QuestionAnswerTemplate
      max_seq_len: 8192

    dataset:
      _component_: torchtune.datasets.multimodal.multimodal_chat_dataset
      source: json
      data_files: data/my_data.json
      split: train
      column_map:
        dialogue: conversations
        image_path: image
      image_dir: /home/user/dataset/
      image_tag: "<image>"
      split: train

Multimodal dataset format
-------------------------

Multimodal datasets are currently expected to follow the :ref:`sharegpt` chat format, where the image paths are in one column
and the user-assistant conversations are in another column.

.. code-block:: text

    |  conversations                     | image        |
    |------------------------------------|--------------|
    | [{"from": "human", "value": "Q1"}, | images/1.jpg |
    |  {"from": "gpt", "value": "A1"}]   |              |

As an example, you can see the schema of the `ShareGPT4V dataset <https://huggingface.co/datasets/Lin-Chen/ShareGPT4V>`_.

Currently, :func:`~torchtune.datasets.multimodal.multimodal_chat_dataset` only supports a single image path per conversation sample.


Loading multimodal datasets from Hugging Face
---------------------------------------------

You simply need to pass in the dataset repo name to ``source``, which is then passed into Hugging Face's ``load_dataset``.
For most datasets, you will also need to specify the ``split`` and/or the subset via ``name``.

.. code-block:: python

    # In code
    from torchtune.models.llama3_2_vision import llama3_2_vision_transform
    from torchtune.datasets.multimodal import multimodal_chat_dataset

    model_transform = llama3_2_vision_transform(
        path="/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model",
        max_seq_len=8192,
        image_size=560,
    )
    ds = multimodal_chat_dataset(
        model_transform=model_transform,
        source="Lin-Chen/ShareGPT4V",
        split="train",
        name="ShareGPT4V",
        image_dir="/home/user/dataset/",
        image_tag="<image>",
    )

.. code-block:: yaml

    # In config
    model_transform:
      _component_: torchtune.models.llama3_2_vision.llama3_2_vision_transform
      path: /tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model
      max_seq_len: 8192
      image_size: 560

    # Tokenizer is passed into the dataset in the recipe
    dataset:
      _component_: torchtune.datasets.multimodal.multimodal_chat_dataset
      source: Lin-Chen/ShareGPT4V
      split: train
      name: ShareGPT4V
      image_dir: /home/user/dataset/
      image_tag: "<image>"

This will use the default column names "conversations" and "image". To change the column names, use the ``column_map`` argument (see :ref:`column_map`).

Loading local and remote multimodal datasets
--------------------------------------------

To load in a local or remote dataset via https that follows the instruct format, you need to specify the ``source``, ``data_files`` and ``split``
arguments. See Hugging Face's ``load_dataset`` `documentation <https://huggingface.co/docs/datasets/main/en/loading#local-and-remote-files>`_
for more details on loading local or remote files. See :ref:`example_multimodal` above.

Loading images
--------------
In many cases, your dataset will contain paths to the images instead of the raw images themselves. :func:`~torchtune.datasets.multimodal.multimodal_chat_dataset`
will automatically handle this for you, but if you are writing a custom message transform for a custom multimodal dataset
(see :ref:`custom_message_transform`), you can use the :func:`~torchtune.data.load_image` utility directly.

.. code-block:: python

    from torchtune.data import load_image
    from pathlib import Path

    sample = {
        "conversations": [
            {
                "from": "human",
                "value": "What time is it on the clock?",
            },
            {
                "from": "gpt",
                "value": "It is 10:00 AM.",
            },
        ],
        "image": "images/clock.jpg",
    }
    image_dir = "/home/user/dataset/"
    pil_image = load_image(Path(image_dir) / Path(sample["image"]))
    print(pil_image)
    # <PIL.Image.Image>

Then, you can add the PIL image directly to the content of the related message. Only PIL images are supported as image content
in :class:`~torchtune.data.Message`, not image paths or urls.

.. code-block:: python

    from torchtune.data import Message

    user_message = None
    for msg in sample["conversations"]:
        if msg["from"] == "human":
            user_message = Message(
                role="user",
                content=[
                    {"type": "image", "content": pil_image},
                    {"type": "text", "content": msg["value"]},
                ]
            )
    print(user_message.contains_media)
    # True
    print(user_message.get_media())
    # [<PIL.Image.Image>]
    print(user_message.text_content)
    # What time is it on the clock?

If the image paths in your dataset are relative paths, you can use the ``image_dir`` parameter in :func:`~torchtune.datasets.multimodal.multimodal_chat_dataset`
to prepend the full path where your images are downloaded locally.

Interleaving images in text
---------------------------
torchtune supports adding multiple images in any locations in the text, as long as your model supports it.

.. code-block:: python

    import PIL
    from torchtune.data import Message

    image_dog = PIL.Image.new(mode="RGB", size=(4, 4))
    image_cat = PIL.Image.new(mode="RGB", size=(4, 4))
    image_bird = PIL.Image.new(mode="RGB", size=(4, 4))

    user_message = Message(
        role="user",
        content=[
            {"type": "image", "content": image_dog},
            {"type": "text", "content": "This is an image of a dog. "},
            {"type": "image", "content": image_cat},
            {"type": "text", "content": "This is an image of a cat. "},
            {"type": "image", "content": image_bird},
            {"type": "text", "content": "This is a bird, the best pet of the three."},
        ]
    )
    print(user_message.contains_media)
    # True
    print(user_message.get_media())
    # [<PIL.Image.Image>, <PIL.Image.Image>, <PIL.Image.Image>]
    print(user_message.text_content)
    # This is an image of a dog. This is an image of a cat. This is a bird, the best pet of the three.

Your dataset may contain image placeholder tags which indicate where in the text the image should be referenced
As an example, see `ShareGPT4V <https://huggingface.co/datasets/Lin-Chen/ShareGPT4V>`, which uses ``"<image>"``.
You can easily create the interleaved message content similar to above with the utility :func:`~torchtune.data.format_content_with_images`,
which replaces the image placeholder tags with the passed in images.

.. code-block:: python

    import PIL
    from torchtune.data import Message, format_content_with_images

    image_dog = PIL.Image.new(mode="RGB", size=(4, 4))
    image_cat = PIL.Image.new(mode="RGB", size=(4, 4))
    image_bird = PIL.Image.new(mode="RGB", size=(4, 4))

    text = "[img]This is an image of a dog. [img]This is an image of a cat. [img]This is a bird, the best pet of the three."
    user_message = Message(
        role="user",
        content=format_content_with_images(
            content=text,
            image_tag="[img]",
            images=[image_dog, image_cat, image_bird],
        ),
    )
    print(user_message.contains_media)
    # True
    print(user_message.get_media())
    # [<PIL.Image.Image>,<PIL.Image.Image>, <PIL.Image.Image>]
    print(user_message.text_content)
    # This is an image of a dog. This is an image of a cat. This is a bird, the best pet of the three.

This is handled automatically for you in :func:`~torchtune.datasets.multimodal.multimodal_chat_dataset` when you pass in
``image_tag``.

Built-in multimodal datasets
----------------------------
- :class:`~torchtune.datasets.multimodal.the_cauldron_dataset`
- :class:`~torchtune.datasets.multimodal.llava_instruct_dataset`
