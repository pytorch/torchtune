# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union
from urllib import request

import torch
import torchvision
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import default_collate, DistributedSampler

from torchtune.data._torchdata import DatasetType, Loader, requires_torchdata
from torchtune.modules.transforms import Transform

from torchtune.utils import get_world_size_and_rank

T = TypeVar("T", bound=type)


def truncate(
    tokens: List[Any],
    max_seq_len: int,
    eos_id: Optional[Any] = None,
    truncation_type: str = "right",
) -> List[Any]:
    """
    Truncate a list of tokens to a maximum length. If eos_id is provided, the last
    token will be replaced with eos_id.

    Args:
        tokens (List[Any]): list of tokens to truncate
        max_seq_len (int): maximum length of the list
        eos_id (Optional[Any]): token to replace the last token with. If None, the
            last token will not be replaced. Default is None.
        truncation_type (str): type of truncation to apply, either "left" or "right".
            Default is "right".

    Returns:
        List[Any]: truncated list of tokens

    Raises:
        ValueError: if truncation_type is not "left" or "right"
    """

    if truncation_type == "left":
        tokens_truncated = tokens[-max_seq_len:]  # Take the last max_seq_len tokens
    elif truncation_type == "right":
        tokens_truncated = tokens[:max_seq_len]  # Take the first max_seq_len tokens
    else:
        raise ValueError(
            f"truncation_type must be 'left' or 'right', got {truncation_type}"
        )

    # Replace the last token with eos_id if necessary
    if eos_id is not None and tokens_truncated and tokens_truncated[-1] != eos_id:
        tokens_truncated[-1] = eos_id

    return tokens_truncated


def load_image(image_loc: Union[Path, str]) -> torch.Tensor:
    """
    Convenience method to load an image in torch.Tensor format from a local file path or remote source.

    Args:
        image_loc (Union[Path, str]): Local file path or remote source pointing to the image
            which will be loaded in PIL format.

    Note:
        If loading an image from a remote source, the function expects the URL provided in ``image_loc``
        to start with "http" or "https" e.g. "https://www.wikipedia.org/en/bird.jpg".

    Raises:
        ValueError:
            If the image cannot be loaded from remote source, **or**
            if the image cannot be opened as a :class:`~torch.Tensor`.

    Examples:
        >>> # Load from remote source
        >>> image = load_image("https://www.wikipedia.org/en/bird.jpg")

        >>> # Load from local file path
        >>> image = load_image(Path("/home/user/bird.jpg"))

    Returns:
        torch.Tensor: The loaded image.
    """
    # If pointing to remote source, try to load to local
    if isinstance(image_loc, str) and image_loc.startswith("http"):
        try:
            image_loc = request.urlopen(image_loc).read()
            image = torchvision.io.decode_image(
                torch.frombuffer(image_loc, dtype=torch.uint8),
                mode="RGB",
            )
        except Exception as e:
            raise ValueError("Failed to load remote image as torch.Tensor") from e

    # Open the local image as a Tensor image
    else:
        try:
            image = torchvision.io.decode_image(image_loc, mode="RGB")
        except Exception as e:
            raise ValueError("Failed to load local image as torch.Tensor") from e
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


def chain(*funcs: List[Callable]) -> Callable:
    """
    Chain a list of functions together into a single function.

    Args:
        *funcs (List[Callable]): list of functions to chain together

    Returns:
        Callable: chained function
    """

    def chained_fn(x):
        for fn in funcs:
            x = fn(x)
        return x

    return chained_fn


@requires_torchdata
def load_hf_dataset(
    source: str,
    transform: Transform,
    filter_fn: Optional[Callable] = None,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    parallel_method: Literal["process", "thread"] = "thread",
    streaming: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> DatasetType:
    """
    Load a HuggingFace dataset (Map or Streaming) and apply a Transform to it.

    Args:
        source (str): HuggingFace dataset source.
        transform (Transform): Transform to apply to the samples of the dataset.
        filter_fn (Optional[Callable]): Filter function to pass to HuggingFace dataset.
        shuffle (bool): Whether to shuffle the dataset. Default is True. For streaming datasets, this is passed to
            HuggingFace dataset as .shuffle(). For map datasets, a DistributedSampler is used.
        seed (int): Seed for the random number generator in the case of Map style dataset shuffling. Default is 0.
        num_workers (int): Number of workers to use for loading the dataset. Default is 0 (no parallelism). Setting this
            greater than 0 will create `parallel_method` workers to perform transforms to the dataset.
        parallel_method (Literal["process", "thread"]): Method to use for parallelism. Default is "thread". No effect if
            num_workers is 0.
        streaming (bool): whether to load a streaming vs map-style dataset. Default False.
        **load_dataset_kwargs (Dict[str, Any]): Additional Keyword arguments to pass to HuggingFace dataset. See Hugging Face's
            documentation.

    Returns:
        A ``torchdata.nodes`` iterator that can be passed directly to a Loader, or combined with other-datasets in a multi-dataset
        sampler.
    """
    from torchdata.nodes import IterableWrapper, ParallelMapper, SamplerWrapper

    if "subset" in load_dataset_kwargs:
        assert (
            "name" not in load_dataset_kwargs
        ), f"found both 'subset' and 'name' found, you may only specify one, {load_dataset_kwargs=}"
        load_dataset_kwargs["name"] = load_dataset_kwargs.pop("subset")
    dataset = load_dataset(source, **load_dataset_kwargs)
    if filter_fn is not None:
        dataset = dataset.filter(filter_fn)

    world_size, rank = get_world_size_and_rank()
    if streaming:
        dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        node = IterableWrapper(dataset)
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        # Note: SamplerWrapper will call set_epoch on the sampler (if defined),
        # and auto-increment the epoch each time the node is reset.
        node = SamplerWrapper(sampler)
        transform = chain(dataset.__getitem__, transform)  # type: ignore

    node = ParallelMapper(
        node, map_fn=transform, num_workers=num_workers, method=parallel_method
    )

    return node


@requires_torchdata
def get_multi_dataset(
    datasets: Dict[str, DatasetType],
    weights: Dict[str, float],
    stop_criteria: str = "CYCLE_UNTIL_ALL_DATASETS_EXHASTED",
    seed: int = 0,
) -> DatasetType:
    """
    Given a dictionary of datasets and their corresponding weights, return a dataset that
    samples from the given datasets according to the specified weights.

    Args:
        datasets (Dict[str, DatasetType]): dictionary of datasets
        weights (Dict[str, float]): dictionary of weights for each dataset. If not
        stop_criteria (str): stop criteria for the sampler. Default "CYCLE_UNTIL_ALL_DATASETS_EXHASTED".
            See also: torchdata.nodes.StopCriteria
        seed (int): seed for the random number generator. Default 0.

    Returns:
        A ``torchdata.nodes`` iterator which can be passed to Loader, or further composed with other Nodes.
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
    collate_fn: Optional[Callable[[Any], Any]] = None,
    drop_last: bool = True,
    num_workers: int = 0,
    parallel_method: Literal["process", "thread"] = "thread",
    prefetch_factor: Optional[int] = 4,
    pin_memory: bool = False,
) -> Loader:
    """
    This will configure TorchData Nodes to approximate torch.utils.data.DataLoader.
    Given a dataset, apply model_transform (eg tokenization), batching, collation,
    memory pinning, and pre-fetching.

    Args:
        dataset (DatasetType): dataset to load. May be a MultiNodeWeightedSampler
        model_transform (Transform): model transform to apply to the samples of the dataset
        batch_size (int): batch size
        collate_fn (Optional[Callable[[Any], Any]]): collate function to apply to the samples of the dataset. If None, use
            torch.utils.data.default_collate. Default None.
        drop_last (bool): whether to drop the last batch. Default is True.
        num_workers (int): number of workers to use for loading the dataset. Default is 0 (no parallelism
        parallel_method (Literal["process", "thread"]): method to use for parallelism. Default is "thread".
        prefetch_factor (Optional[int]): number of batches to prefetch. Default is 4.
        pin_memory (bool): whether to pin memory. Default is False.

    Returns:
        A ``torchdata.nodes`` Loader, an Iterable that returns batches.
    """

    from torchdata.nodes import Batcher, ParallelMapper, PinMemory, Prefetcher

    if collate_fn is None:
        collate_fn = default_collate

    node = ParallelMapper(
        dataset, map_fn=model_transform, num_workers=num_workers, method=parallel_method
    )
    node = Batcher(node, batch_size, drop_last=drop_last)
    node = ParallelMapper(
        node, map_fn=collate_fn, num_workers=num_workers, method=parallel_method
    )
    if pin_memory:
        node = PinMemory(node)
    if prefetch_factor is not None:
        node = Prefetcher(node, prefetch_factor)

    return Loader(node)
