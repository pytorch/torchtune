.. _model_transform_usage_label:

=====================
Multimodal Transforms
=====================

Multimodal model transforms apply model-specific data transforms to each modality and prepares :class:`~torchtune.data.Message`
objects to be input into the model. torchtune currently supports text + image model transforms.
These are intended to be drop-in replacements for tokenizers in multimodal datasets and support the standard
``encode``, ``decode``, and ``tokenize_messages``.

.. code-block:: python

    # torchtune.models.llama3_2_vision.Llama3VisionTransform
    class Llama3VisionTransform(ModelTokenizer, Transform):
        def __init__(...):
            # Text transform - standard tokenization
            self.tokenizer = llama3_tokenizer(...)
            # Image transforms
            self.transform_image = CLIPImageTransform(...)
            self.xattn_mask = VisionCrossAttentionMask(...)


.. code-block:: python

    from torchtune.models.llama3_2_vision import Llama3VisionTransform
    from torchtune.data import Message
    from PIL import Image

    sample = {
        "messages": [
            Message(
                role="user",
                content=[
                    {"type": "image", "content": Image.new(mode="RGB", size=(224, 224))},
                    {"type": "image", "content": Image.new(mode="RGB", size=(224, 224))},
                    {"type": "text", "content": "What is common in these two images?"},
                ],
            ),
            Message(
                role="assistant",
                content="A robot is in both images.",
            ),
        ],
    }
    transform = Llama3VisionTransform(
        path="/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model",
        tile_size=224,
        patch_size=14,
    )
    tokenized_dict = transform(sample)
    print(transform.decode(tokenized_dict["tokens"], skip_special_tokens=False))
    # '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|><|image|>What is common in these two images?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nA robot is in both images.<|eot_id|>'
    print(tokenized_dict["encoder_input"]["images"][0].shape)  # (num_tiles, num_channels, tile_height, tile_width)
    # torch.Size([4, 3, 224, 224])


Using model transforms
----------------------
You can pass them into any multimodal dataset builder just as you would a model tokenizer.

.. code-block:: python

    from torchtune.datasets.multimodal import the_cauldron_dataset
    from torchtune.models.llama3_2_vision import Llama3VisionTransform

    transform = Llama3VisionTransform(
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
Model transforms are expected to process both text and images in the sample dictionary.
Both should be contained in the ``"messages"`` field of the sample.

The following methods are required on the model transform:

- ``tokenize_messages``
- ``__call__``

.. code-block:: python

    from torchtune.modules.tokenizers import ModelTokenizer
    from torchtune.modules.transforms import Transform

    class MyMultimodalTransform(ModelTokenizer, Transform):
        def __init__(...):
            self.tokenizer = my_tokenizer_builder(...)
            self.transform_image = MyImageTransform(...)

        def tokenize_messages(
            self,
            messages: List[Message],
            add_eos: bool = True,
        ) -> Tuple[List[int], List[bool]]:
            # Any other custom logic here
            ...

            return self.tokenizer.tokenize_messages(
                messages=messages,
                add_eos=add_eos,
            )

        def __call__(
            self, sample: Mapping[str, Any], inference: bool = False
        ) -> Mapping[str, Any]:
            # Expected input parameters for vision encoder
            encoder_input = {"images": [], "aspect_ratio": []}
            messages = sample["messages"]

            # Transform all images in sample
            for message in messages:
                for image in message.get_media():
                    out = self.transform_image({"image": image}, inference=inference)
                    encoder_input["images"].append(out["image"])
                    encoder_input["aspect_ratio"].append(out["aspect_ratio"])
            sample["encoder_input"] = encoder_input

            # Transform all text - returns same dictionary with additional keys "tokens" and "mask"
            sample = self.tokenizer(sample, inference=inference)

            return sample

    transform = MyMultimodalTransform(...)
    sample = {
        "messages": [
            Message(
                role="user",
                content=[
                    {"type": "image", "content": Image.new(mode="RGB", size=(224, 224))},
                    {"type": "image", "content": Image.new(mode="RGB", size=(224, 224))},
                    {"type": "text", "content": "What is common in these two images?"},
                ],
            ),
            Message(
                role="assistant",
                content="A robot is in both images.",
            ),
        ],
    }
    tokenized_dict = transform(sample)
    print(tokenized_dict)
    # {'encoder_input': {'images': ..., 'aspect_ratio': ...}, 'tokens': ..., 'mask': ...}


Example model transforms
------------------------
- Llama 3.2 Vision
    - :class:`~torchtune.models.llama3_2_vision.Llama3VisionTransform`
