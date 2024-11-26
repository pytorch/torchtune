# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union
from urllib import request

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

from torch.utils.data import DistributedSampler

from torchtune.data._torchdata import DatasetType, Loader, requires_torchdata
from torchtune.modules.transforms import Transform

T = TypeVar("T", bound=type)


def truncate(
    tokens: List[Any],
    max_seq_len: int,
    eos_id: Optional[Any] = None,
) -> List[Any]:
    """
    Truncate a list of tokens to a maximum length. If eos_id is provided, the last
    token will be replaced with eos_id.

    Args:
        tokens (List[Any]): list of tokens to truncate
        max_seq_len (int): maximum length of the list
        eos_id (Optional[Any]): token to replace the last token with. If None, the
            last token will not be replaced. Default is None.

    Returns:
        List[Any]: truncated list of tokens
    """
    tokens_truncated = tokens[:max_seq_len]
    if eos_id is not None and tokens_truncated[-1] != eos_id:
        tokens_truncated[-1] = eos_id
    return tokens_truncated


def load_image(image_loc: Union[Path, str]) -> "PIL.Image.Image":
    """
    Convenience method to load an image in PIL format from a local file path or remote source.

    Args:
        image_loc (Union[Path, str]): Local file path or remote source pointing to the image
            which will be loaded in PIL format.

    Note:
        If loading an image from a remote source, the function expects the URL provided in ``image_loc``
        to start with "http" or "https" e.g. "https://www.wikipedia.org/en/bird.jpg".

    Raises:
        ValueError: If the image cannot be loaded from remote source.
        ValueError: If the image cannot be opened as a :class:`~PIL.Image.Image`.

    Examples:
        >>> # Load from remote source
        >>> image = load_image("https://www.wikipedia.org/en/bird.jpg")

        >>> # Load from local file path
        >>> image = load_image(Path("/home/user/bird.jpg"))

    Returns:
        PIL.Image.Image: The loaded image.
    """
    # Hackily import PIL to avoid burdensome import in the main module
    # TODO: Fix this
    from PIL import Image

    # If pointing to remote source, try to load to local
    if isinstance(image_loc, str) and image_loc.startswith("http"):
        try:
            image_loc = request.urlopen(image_loc)
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_loc}") from e

    # Open the local image as a PIL image
    try:
        image = Image.open(image_loc)
    except Exception as e:
        raise ValueError(f"Failed to open image as PIL Image from {image_loc}") from e

    return image


def format_content_with_images(
    content: str, *, image_tag: str, images: List["PIL.Image.Image"]
) -> List[Dict[str, Any]]:
    """
    Given a raw text string, split by the specified ``image_tag``
    and form into list of dictionaries to be used in the :class:`~torchtune.data.Message` content
    field::

        [
            {
                "role": "system" | "user" | "assistant",
                "content":
                    [
                        {"type": "image", "content": <PIL.Image.Image>},
                        {"type": "text", "content": "This is a sample image."},
                    ],
            },
            ...
        ]

    Args:
        content (str): raw message text
        image_tag (str): string to split the text by
        images (List["PIL.Image.Image"]): list of images to be used in the content

    Raises:
        ValueError: If the number of images does not match the number of image tags in the content

    Examples:
        >>> content = format_content_with_images(
        ...     "<|image|>hello <|image|>world",
        ...     image_tag="<|image|>",
        ...     images=[<PIL.Image.Image>, <PIL.Image.Image>]
        ... )
        >>> print(content)
        [
            {"type": "image", "content": <PIL.Image.Image>},
            {"type": "text", "content": "hello "},
            {"type": "image", "content": <PIL.Image.Image>},
            {"type": "text", "content": "world"}
        ]

    Returns:
        List[Dict[str, Any]]: list of dictionaries to be used in the :class:`~torchtune.data.Message` content field
    """
    num_image_tags_in_content = content.count(image_tag)
    if len(images) != num_image_tags_in_content:
        raise ValueError(
            f"Number of images ({len(images)}) does not match number of image tags "
            f"({num_image_tags_in_content}) in content: {content}"
        )

    split_content = content.split(image_tag)
    final_content_list = []
    for i, substr in enumerate(split_content):
        if len(substr) > 0:
            final_content_list.append({"type": "text", "content": substr})
        if i < len(split_content) - 1:
            final_content_list.append({"type": "image", "content": images.pop(0)})

    return final_content_list


@requires_torchdata
def load_hf_dataset(
    source: str,
    transform: Transform,
    filter_fn: Optional[Callable] = None,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 1,
    parallel_method: Literal["process", "thread"] = "thread",
    **load_dataset_kwargs: Dict[str, Any],
) -> DatasetType:
    from torchdata.nodes import IterableWrapper, Mapper, ParallelMapper, SamplerWrapper

    # Need to lazy import to avoid circular dependency
    from torchtune.training._distributed import get_world_size_and_rank

    streaming = load_dataset_kwargs.get("streaming", False)
    if "subset" in load_dataset_kwargs:
        assert (
            "name" not in load_dataset_kwargs
        ), f"found both 'subset' and 'name' found, you may only specify one, {load_dataset_kwargs=}"
        load_dataset_kwargs["name"] = load_dataset_kwargs.pop("subset")
    dataset = load_dataset(source, **load_dataset_kwargs)
    if filter_fn is not None:
        dataset = dataset.filter(filter_fn)

    if num_workers == 0:
        _Mapper = Mapper  # type: ignore
    else:
        _Mapper = functools.partial(
            ParallelMapper,  # type: ignore
            num_workers=num_workers,
            method=parallel_method,
        )
    world_size, rank = get_world_size_and_rank()
    if streaming:
        dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        node = IterableWrapper(dataset)  # type: ignore
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        node = SamplerWrapper(sampler)  # type: ignore
        node = _Mapper(node, map_fn=dataset.__getitem__)

    node = _Mapper(node, transform)

    return node


@requires_torchdata
def get_multi_dataset(
    datasets: dict[str, DatasetType],
    weights: dict[str, float],
    stop_criteria: str = "CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED",
    seed: int = 0,
) -> DatasetType:
    """
    Given a dictionary of datasets and their corresponding weights, return a dataset that
    samples from the given datasets according to the specified weights.

    Args:
        datasets (Dict[str, Any]): dictionary of datasets
        weights (Optional[Dict[str, float]]): dictionary of weights for each dataset. If not

    """
    from torchdata.nodes import MultiNodeWeightedSampler

    return MultiNodeWeightedSampler(
        source_nodes=datasets,
        weights=weights,
        stop_criteria=stop_criteria,
        seed=seed,
    )


@requires_torchdata
def get_dataloader(
    dataset: DatasetType,
    model_transform: Transform,
    batch_size: int,
    collate_fn: Callable[[Any], Any],
    packed: bool = False,
    drop_last: bool = True,
    num_workers: int = 0,
    parallel_method: Literal["process", "thread"] = "thread",
    prefetch_factor: Optional[int] = 4,
    pin_memory: bool = False,
) -> Loader:
    if packed:
        raise ValueError("Multimodal datasets don't support packing yet.")

    from torchdata.nodes import Batcher, Mapper, ParallelMapper, PinMemory, Prefetcher

    if num_workers == 0:
        _Mapper = Mapper  # noqa[N806]
    else:
        _Mapper = functools.partial(  # noqa[N806]
            ParallelMapper,
            num_workers=num_workers,
            method=parallel_method,
        )

    node = _Mapper(dataset, map_fn=model_transform)
    node = Batcher(node, batch_size, drop_last=drop_last)
    node = _Mapper(node, map_fn=collate_fn)
    if pin_memory:
        node = PinMemory(node)
    if prefetch_factor is not None:
        node = Prefetcher(node, prefetch_factor)

    return Loader(node)
