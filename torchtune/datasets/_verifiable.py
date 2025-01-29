from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import ModelTokenizer
from torchtune.data import Message


class PromptToMessage(Transform):
    """
    Message transform class that converts a single sample with "prompt" and other fields,
    (or equivalent fields specified in column_map) to a user message.

        |  prompt         |  output          |
        |-----------------|------------------|
        | "user prompt"   | "other data" |

    Args:
        train_on_input (bool): Whether the model is trained on the user prompt or not.
            Default is False.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "input"
            and "output" column names to the actual column names in the dataset. Keys should
            be "input" and "output" and values should be the actual column names. Default is None,
            keeping the default "input" and "output" column names.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Default is None.
        image_dir (Optional[Path]): path to the directory containing the images that is prepended to all image
            paths in the dataset. For example, if ``image_dir="/home/user/dataset/"` and the sample image path
            was ``"images/1.jpg"``, the final image path that will be loaded is ``"/home/user/dataset/images/1.jpg"``.
            If None, assume images are available in current working directory or are located
            on a remote url. For text-only, leave as None. Default is None.

    Raises:
        ValueError:
            If ``column_map`` is provided and ``input`` not in ``column_map``, or
                ``output`` not in ``column_map``, **or**
            if ``image_dir`` is provided but ``image`` not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: bool = False,
        column_map: Optional[Dict[str, str]] = None,
        new_system_prompt: Optional[str] = None
    ):
        self.train_on_input = train_on_input
        self.new_system_prompt = new_system_prompt

        self.column_map = column_map

        if self.column_map is not None:
            if "prompt" not in self.column_map:
                raise ValueError(
                    f"Expected a key of 'input' in column_map but found {self.column_map.keys()}."
                )
        else:
            self.column_map = {"prompt": "prompt"}

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        content = [{"type": "text", "content": sample[self.column_map["prompt"]]}]


        messages = [
            Message(
                role="user",
                content=content,
                masked=not self.train_on_input,
                eot=True,
            ),
        ]
        if self.new_system_prompt is not None:
            messages = [
                Message(
                    role="system", content=self.new_system_prompt, masked=True, eot=True
                )
            ] + messages
        return {"messages": messages}


class VerifiableDataset(Dataset):
    """
    Class for fine-tuning with verifiable rewards. This class supplies a prompt, and 
    maintains other columns to be passed through to the verifier. 

    At a high level, this class will load the data from source and apply the following pre-processing steps when a
    sample is retrieved:

    1. Dataset-specific transform. This is typically unique to each dataset and extracts
       the necessary prompt and chosen/rejected columns into torchtune's :class:`~torchtune.data.Message`
       format, a standardized API for all model tokenizers.
    2. Tokenization with optional prompt template if configured


    All datasets are formatted into a list of :class:`~torchtune.data.Message`

    The :class:`~torchtune.data.Message` forms the core data unit that all tokenizer
    APIs expect. The key component of this class that ensures any dataset is transformed
    into this format is the ``message_transform``. This is a callable class that takes
    in a sample dictionary - typically a single row from the source dataset - that
    processes the sample in any configurable way to output a list of messages::

        [
            Message(
                role=<system|user|assistant|ipython>,
                content=<message>,
            ),
            ...
        ]

    For any custom dataset, use the ``message_transform`` to contain all pre-processing to
    return the list of messages.

    Args:
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        message_transform (Transform): callable that keys into the desired fields in the sample
            and converts text content to a list of :class:`~torchtune.data.Message`. It is expected that the final list
            of messages are stored in the ``"chosen"`` and ``"rejected"`` keys.
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
            Since PreferenceDataset only supports text data, it requires a
            :class:`~torchtune.modules.transforms.tokenizers.ModelTokenizer` instead of the ``model_transform`` in
            :class:`~torchtune.datasets.SFTDataset`.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False. Packed is
            currently not supported for ``VerifiableDataset`` and a ``ValueError`` will be raised if this is set to True.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.

    Raises:
        ValueError: If ``packed`` is True, this feature is not supported for ``VerifiableDataset``.
    """

    def __init__(
        self,
        *,
        source: str,
        message_transform: Transform,
        tokenizer: ModelTokenizer,
        filter_fn: Optional[Callable] = None,
        packed: bool = False,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        if packed:
            raise ValueError(
                "Packed is currently not supported for verofiable datasets."
            )

        self._tokenizer = tokenizer
        self._message_transform = message_transform
        self._data = load_dataset(source, **load_dataset_kwargs)

        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        transformed_sample = self._message_transform(sample)

        input_ids, masks = self._tokenizer.tokenize_messages(
            transformed_sample["messages"],
        )
        labels = list(
            np.where(masks, CROSS_ENTROPY_IGNORE_IDX, input_ids)
        )

        assert len(input_ids) == len(labels)
        
        tokenized_dict = dict(
            tokens=input_ids,
            labels=labels,
            raw=sample,
        )

        return tokenized_dict


def verifiable_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> VerifiableDataset:
    """
    Configures a custom  dataset for use with verifiable rewards. 

    This builder function can be used to configure a custom preference dataset directly from the yaml config, 
    as it is made to be config friendly.

    This function requires the dataset to have a "prompt" column. Other columns will be maintained for access
    by the verifier. 


    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is:
    set to ``False`` by default.

    - If ``train_on_input`` is True, the prompt is used during training and
      contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100).

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text"), pass
            in the filepath in ``data_files``, and set ``split="train"``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns "chosen" and "rejected"
            in the message transform :class:`~torchtune.data.ChosenRejectedToMessages` to the new column names in
            the dataset. Keys should be "chosen" and "rejected" and values should be the actual column names.
            If None, keep the default columns "chosen" and "rejected".
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        new_system_prompt (Optional[str]): if specified, prepend a system message to every sample for both chosen
            and rejected. This can serve as instructions to guide the model response. Setting this will OVERRIDE
            any system messages already present in the dataset. Default is None.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Examples:

    ::

        my_verifiable_dataset.json
        [
            {
                "prompt": [
                    {
                        "content": "If jack has 10 apples and gives 3 away, how many apples does he have?",
                        "role": "user"
                    },
                ],
               "answer": "7"
            }
        ]


    Returns:
        VerifiableDataset: The preference dataset built from source paired data.
    """

    message_transform = PromptToMessage(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )

    return VerifiableDataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )