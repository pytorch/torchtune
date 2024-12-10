# Overview

This is an RFC regarding how we should support datasets for finetuning text-conditioned image generation models.

A basic data pipeline for this would be:
1. Load the JSON/CSV/TSV/Parquet/LMDB/etc. file containing the image paths/urls and captions
2. For each pair:
   - load/download the image
   - resize the image and optionally randomly augment it (horizontal flip, etc.) and normalize it
   - optionally randomly augment the caption (rearrange caption parts, etc.)
   - tokenize the caption using the model's tokenizer
3. collate into a batch

At a broad level, this fits well into our current TorchTune data ecosystem (except we wouldn't use the "list of Message objects" abstraction, which would change how we interact with the model's tokenizer).

In TorchTune, a simple version would look something like this:

```yaml
dataset:
    _component_: torchtune.datasets.img_caption_dataset
    path: ~/my_dataset/data.tsv
    img_transform:
        resize: [256, 256]
        center_crop: true
        horizontal_flip: 0.5
    caption_transform:
        drop: 0.05
        shuffle_parts: 0.1
tokenizer:
    _component_: torchtune.models.flux.FluxTransform
    clip_tokenizer_path: ...
    t5_tokenizer_path: ...
    t5_max_seq_len: 256
```

```python
def img_caption_dataset(
    model_transform: Transform,
    *,
    path: str,
    img_transform: Config,
    caption_transform: Config,
):
    """Builder for an image caption dataset."""
    data = _load_img_text_dataset(path)
    img_transform = _build_torchvision_transforms(img_transform)
    caption_transform = _CaptionTransform(caption_transform)
    return ImgTextDataset(
        data,
        img_transform=img_transform,
        text_tranform=caption_transform,
        model_transform=model_transform,
    )


def _load_img_text_dataset(path):
    if '.' not in path:
        return datasets.load_dataset(path, ...)

    path = Path(path).expanduser().resolve()
    if path.suffix == ".tsv":
        data = []
        with open(path, "r") as f:
            for line in f:
                img_path_or_url, text = [x.strip() for x in line.split("\t")]
                data.append((img_path_or_url, text))
        return data

    elif path.suffix == "...":
        ...


def _build_torchvision_transforms(cfg):
    """
    Create a series of torchvision transforms
    (resize, crop, flip, etc.)
    """
    ...


class _CaptionTransform:
    """
    Callable that randomly augments image captions with comma-separated parts
    (shuffle parts, randomly drop entire caption, etc.)
    (or does nothing if disabled)
    """

    def __init__(self, cfg): ...

    def __call__(self, caption: str) -> str: ...


class ImgTextDataset(torch.utils.data.Dataset):
    def __init__(self, data, img_transform, text_transform, model_transform):
        self._data = data
        self._img_transform = img_transform
        self._text_transform = text_transform
        self._model_transform = model_transform

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        img_path_or_url, text = self._data[idx]
        img = (
            Image.open(BytesIO(requests.get(img_path_or_url).content))
            if img_path_or_url.startswith(("http://", "https://", "ftp://", "ftps://"))
            else Image.open(img_path_or_url)
        )
        img = self._img_transform(img)
        text = self._text_transform(text)
        data_dict = self._model_transform(img, text)
        return data_dict


class FluxTransform(Transform):
    def __init__(self, clip_tokenizer_path, t5_tokenizer_path, t5_max_seq_len):
        ...

    def __call__(self, img, text):
        return {
            'img': (img / 127.5) - 1.0,
            'clip_text_tokens': self._clip_tokenizer(text),
            't5_text_tokens': self._t5_tokenizer(text),
        }
```

# TODO: Collate

We'll need to generalize our collate functions such that they can handle data outside of the tokens-and-labels format they currently expect. I will update this section after I've looked into this.

# Caching/Preprocessing

From what I've seen online, some people finetune image generators on massive datasets, but most people just finetune on very small personal datasets, often 5-100 images. So we should probably add support for various caching/preprocessing options that increase disk/mem usage in order to achieve faster iterations. Some ideas for optional configurations:

- cache up to N images in each data worker so they don't have to load them fresh from disk each epoch
- in the extreme case of like <10 images, we could even just keep the whole dataset on each GPU so we don't have to transfer them each step
- in the case of a web dataset, save up to N downloaded images to local storage for the next epoch
- before training, preprocess the outputs of frozen parts of the model (text tokens, image autoencoder embeddings) and save them to disk so that we don't have to recompute every epoch
   - tokenization would be negligible but I bet preprocessing the Flux image encoding would save a lot of time and GPU memory
   - this could also be done on the fly, i.e. caching instead of preprocessing. During the first epoch, save the intermediate values to disk and reuse them in all the next epochs. But this makes the code much more complicated.

But we should evaluate whether each of these is worth it:
- how much performance gain would you actually get? and under what circumstances?
- how much would it complicate the code and the configs?

# Dataset Creation

Should we include scripts/utilities for creating the captions? Users will probably often have just a folder with a bunch of images that they want to finetune on. So we could help them turn that folder into a dataset by using some model to automatically caption them. We could even provide our own models for this by distilling the image captioning capabilities of Llama3.2V-90B into several smaller Llama3.2V models, and let the user pick the one that fits on their device.

We'll also want to support adding words/phrases to the caption that tell the model to generate in the style of this dataset. For example, if I'm finetuning a model on images of myself, I'll want to include something like "a photo of cpelletier" in the caption so that the model learns to associate "cpelletier" with my face.

# User Experience

- Regarding loading the TSV/Parquet/whatever data file, should we just rely on huggingface's `load_dataset` like we currently do in `SFTDataset`? It keeps the code simpler, but it makes the user leave torchtune and go read the huggingface docs, which is overkill if they just have some simple JSON file we could easily load ourselves.
- In addition to absolute image paths in the data file, we should probably support image paths relative to the dataset folder, because it would be super annoying if you had to regenerate your data file any time to move the dataset to a new location.
- There's currently some potentially unnecessary fields in the config. For example with Flux models, the model determines the image size and the T5 tokenizer sequence length. Is it better to pass this information to the image transform and model transform, respectively? Which complicates the code but lowers the chance of user error. Or is it better to have the user define these values in the dataset config and tokenizer config, respectively? Which puts the burden on the user to match what the model expects.
- Should we add scripts/utilities for inspecting the dataset? It's nice to see a preview of what a batch looks like, especially when you're messing around with color jitter and other hard-to-configure image augmentations.

# Other
- Naming of the image-text dataset builders/classes? Maybe the more verbose `image_caption_dataset_for_image_generation` is better to make it clear that this is NOT for something like finetuning a VLM to do image captioning (although maybe it could be generalized to the point where it can also do lists of Message objects and therefore can be used for whatever purpose).
- Support multiple captions per image? I can imagine people wanting to generate multiple captions for their images, and randomly selecting one at a time during training to prevent overfitting. It's kinda a caption augmentation but it's unique for each caption so it would have to be supported at the data level.
