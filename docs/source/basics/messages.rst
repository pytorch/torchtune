.. _messages_usage_label:

========
Messages
========

Messages are a core component in torchtune that govern how text and multimodal content is tokenized. It serves as the common interface
for all tokenizer and datasets APIs to operate on. Messages contain information about the text content, which role is sending the text
content, and other information relevant for special tokens in model tokenizers. For more information about the individual parameters
for Messages, see the API ref for :class:`~torchtune.data.Message`. Here, we briefly discuss how to create messages, format messages, and
tokenize messages.


Creating Messages
-----------------

Messages can be created via the standard class constructor or directly from a dictionary.

.. code-block:: python

    from torchtune.data import Message

    msg = Message(
        role="user",
        content="Hello world!",
        masked=True,
        eot=True,
        ipython=False,
    )
    # This is identical
    msg = Message.from_dict(
        {
            "role": "user",
            "content": "Hello world!",
            "masked": True,
            "eot": True,
            "ipython": False,
        },
    )
    print(msg.content)
    # [{'type': 'text', 'content': 'Hello world!'}]

Content is formatted as a list of dictionaries. This is because Messages can also contain multimodal content, such as images.

Images in Messages
^^^^^^^^^^^^^^^^^^
For multimodal datasets, you need to add the image as a :class:`~PIL.Image.Image` to the corresponding :class:`~torchtune.data.Message`.
To add it to the beginning of the message, simply prepend it to the content list.

.. code-block:: python

    import PIL
    from torchtune.data import Message

    img_msg = Message(
        role="user",
        content=[
            {
                "type": "image",
                # Place your image here
                "content": PIL.Image.new(mode="RGB", size=(4, 4)),
            },
            {"type": "text", "content": "What's in this image?"},
        ],
    )

This will indicate to the model tokenizers where to add the image special token and will be processed by the model transform
appropriately.

In many cases, you will have an image path instead of a raw :class:`~PIL.Image.Image`. You can use the :func:`~torchtune.data.load_image`
utility for both local paths and remote paths.

.. code-block:: python

    import PIL
    from torchtune.data import Message, load_image

    image_path = "path/to/image.jpg"
    img_msg = Message(
        role="user",
        content=[
            {
                "type": "image",
                # Place your image here
                "content": load_image(image_path),
            },
            {"type": "text", "content": "What's in this image?"},
        ],
    )

If your dataset contain image tags, or placeholder text to indicate where in the text the image should be inserted,
you can use the :func:`~torchtune.data.format_content_with_images` to split the text into the correct content list
that you can pass into the content field of Message.

.. code-block:: python

    import PIL
    from torchtune.data import format_content_with_images

    content = format_content_with_images(
        "<|image|>hello <|image|>world",
        image_tag="<|image|>",
        images=[PIL.Image.new(mode="RGB", size=(4, 4)), PIL.Image.new(mode="RGB", size=(4, 4))]
    )
    print(content)
    # [
    #     {"type": "image", "content": <PIL.Image.Image>},
    #     {"type": "text", "content": "hello "},
    #     {"type": "image", "content": <PIL.Image.Image>},
    #     {"type": "text", "content": "world"}
    # ]

Message transforms
^^^^^^^^^^^^^^^^^^
Message transforms are convenient utilities to format raw data into a list of torchtune :class:`~torchtune.data.Message`
objects. See :ref:`message_transform_usage_label` for more discussion.


Formatting messages with prompt templates
-----------------------------------------

Prompt templates provide a way to format messages into a structured text template. You can simply call any class that inherits
from :class:`~torchtune.data.PromptTemplateInterface` on a list of Messages and it will add the appropriate text to the content
list.

.. code-block:: python

    from torchtune.models.mistral import MistralChatTemplate
    from torchtune.data import Message

    msg = Message(
        role="user",
        content="Hello world!",
        masked=True,
        eot=True,
        ipython=False,
    )
    template = MistralChatTemplate()
    templated_msg = template([msg])
    print(templated_msg[0].content)
    # [{'type': 'text', 'content': '[INST] '},
    # {'type': 'text', 'content': 'Hello world!'},
    # {'type': 'text', 'content': ' [/INST] '}]
