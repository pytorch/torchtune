.. _model_transform_usage_label:

=====================
Multimodal Transforms
=====================

Multimodal model transforms apply model-specific data transforms to each modality and prepares :class:`~torchtune.data.Message`
objects to be input into the model. torchtune currently supports text + image model transforms.
These are intended to be drop-in replacements for tokenizers in multimodal datasets and support the standard
``encode``, ``decode``, and ``tokenize_messages``.

.. code-block:: python

    # torchtune.models.flamingo.FlamingoTransform
    class FlamingoTransform(ModelTokenizer, Transform):
        def __init__(...):
            # Text transform - standard tokenization
            self.tokenizer = llama3_tokenizer(...)
            # Image transforms
            self.transform_image = CLIPImageTransform(...)
            self.xattn_mask = VisionCrossAttentionMask(...)


.. code-block:: python

    from torchtune.models.flamingo import FlamingoTransform
    from torchtune.data import Message
    from PIL import Image

    sample = {
        "messages": [
            Message(
                role="user",
                content=[
                    {"type": "image", "content": Image.new(mode="RGB", size=(224, 224))},
                    {"type": "text", "content": "What is in this image?"},
                ],
            ),
            Message(
                role="assistant",
                content="A robot.",
            ),
        ],
    }
    transform = FlamingoTransform(
        path="/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model",
        tile_size=224,
        patch_size=14,
    )
    tokenized_dict = transform(sample)
    print(transform.decode(tokenized_dict["tokens"], skip_special_tokens=False))
    # '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>What is in this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nA robot.<|eot_id|>'
    print(tokenized_dict["encoder_input"]["images"][0].shape)  # (num_tiles, num_channels, tile_height, tile_width)
    # torch.Size([4, 3, 224, 224])


Using model transforms
----------------------
You can pass them into any multimodal dataset builder just as you would a model tokenizer.

.. code-block:: python

    from torchtune.datasets.multimodal import the_cauldron_dataset
    from torchtune.models.flamingo import FlamingoTransform

    transform = FlamingoTransform(
        path="/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model",
        tile_size=224,
        patch_size=14,
    )
    ds = the_cauldron_dataset(
        model_transform=transform,
        subset="ai2d",
    )
    tokenized_dict = ds[0]
    print(transform.decode(tokenized_dict["tokens"], skip_special_tokens=False))
    # <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    #
    # <|image|>Question: What do respiration and combustion give out
    # Choices:
    # A. Oxygen
    # B. Carbon dioxide
    # C. Nitrogen
    # D. Heat
    # Answer with the letter.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    #
    # Answer: B<|eot_id|>
    print(tokenized_dict["encoder_input"]["images"][0].shape)  # (num_tiles, num_channels, tile_height, tile_width)
    # torch.Size([4, 3, 224, 224])

Creating model transforms
-------------------------
Model transforms are expected to


Example model transforms
------------------------
- Flamingo
    - :class:`~torchtune.models.flamingo.FlamingoTransform`
