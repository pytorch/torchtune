# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import re
import sys
import unittest
from contextlib import contextmanager
from functools import partial
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Mapping, Optional, TextIO, Tuple, Union

import pytest

import torch
from torch import nn
from torchtune.data import Message, PromptTemplate, truncate
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform

skip_if_cuda_not_available = unittest.skipIf(
    not torch.cuda.is_available(), "CUDA is not available"
)

CKPT_MODEL_PATHS = {
    "llama2_tune": "/tmp/test-artifacts/small-ckpt-tune-03082024.pt",
    "llama2_meta": "/tmp/test-artifacts/small-ckpt-meta-03082024.pt",
    "llama2_hf": "/tmp/test-artifacts/small-ckpt-hf-03082024.pt",
    "llama2_reward_hf": "/tmp/test-artifacts/small-ckpt-hf-reward-07122024.pt",
    "llama3_tune": "/tmp/test-artifacts/small-ckpt-tune-llama3-05052024.pt",
    "llama2_7b": "/tmp/test-artifacts/llama2-7b-torchtune.pt",
}

TOKENIZER_PATHS = {
    "llama2": "/tmp/test-artifacts/tokenizer.model",
    "llama3": "/tmp/test-artifacts/tokenizer_llama3.model",
}

# Taken from Open-Orca/SlimOrca-Dedup on Hugging Face:
# https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup
CHAT_SAMPLE = {
    "system": "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",  # noqa: B950
    "user": "Please briefly summarize this news article:\n\nAOL.com Video - Father Lets 8-Year-Old Drive On Icy Road\n\nDescription:Would you let your 8-year-old drive your car? How about on an icy road? Well one father in Russia did just that, and recorded the entire thing. To her credit, the child seemed to be doing a great job. (0:44)\n\nTags: 8-year-old driver , caught on camera , child driver , pix11\n\nSummary:",  # noqa: B950
    "assistant": "A father in Russia allowed his 8-year-old child to drive his car on an icy road and recorded the event. The child appeared to be handling the situation well, showcasing their driving skills despite the challenging conditions.",  # noqa: B950
}

MESSAGE_SAMPLE_TRAIN_ON_INPUT = [
    Message(
        role="system",
        content=CHAT_SAMPLE["system"],
    ),
    Message(
        role="user",
        content=CHAT_SAMPLE["user"],
    ),
    Message(
        role="assistant",
        content=CHAT_SAMPLE["assistant"],
    ),
]

MESSAGE_SAMPLE = [
    Message(role="system", content=CHAT_SAMPLE["system"], masked=True),
    Message(role="user", content=CHAT_SAMPLE["user"], masked=True),
    Message(
        role="assistant",
        content=CHAT_SAMPLE["assistant"],
    ),
]


class DummyTokenizer(ModelTokenizer, Transform):
    def __init__(self, max_seq_len: Optional[int] = None):
        self.max_seq_len = max_seq_len

    def encode(self, text, add_bos=True, add_eos=True, **kwargs) -> List[int]:
        words = text.split()
        tokens = [len(word) for word in words]
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def tokenize_messages(
        self,
        messages: List[Message],
    ) -> Tuple[List[int], List[bool]]:
        """
        A simplified version of Llama2Tokenizer's ``tokenize_messages`` for testing purposes.
        """
        start_of_turn = True
        end_of_turn = False
        tokenized_messages = []
        mask = []
        for message in messages:
            # If assistant message, this is the end of a turn
            end_of_turn = message.role == "assistant"

            # Prepend BOS on start of new turns
            if start_of_turn:
                tokenized_messages.append(self.bos_id)
                mask.append(message.masked)

            # Tokenize current message, append with masks
            tokens = []
            for item in message.content:
                if item["type"] == "text":
                    tokens = tokens + self.encode(
                        item["content"],
                        add_bos=False,
                        add_eos=False,
                    )
                elif item["type"] == "image":
                    tokens = tokens + [self.image_id]

            tokenized_messages.extend(tokens)
            mask.extend([message.masked] * len(tokens))

            # If assistant message, append EOS at end
            if end_of_turn:
                tokenized_messages.append(self.eos_id)
                mask.append(message.masked)
                end_of_turn = False
                start_of_turn = True
            else:
                start_of_turn = False

            # Break out early if we reach max_seq_len
            if self.max_seq_len and len(tokenized_messages) >= self.max_seq_len:
                break

        # Finally, truncate if necessary
        if self.max_seq_len:
            tokenized_messages = truncate(
                tokenized_messages, self.max_seq_len, self.eos_id
            )
            mask = truncate(mask, self.max_seq_len, message.masked)

        return tokenized_messages, mask

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        messages: List[Message] = sample.pop("messages")
        images = []
        for message in messages:
            images += message.get_media()
        tokens, mask = self.tokenize_messages(messages)
        sample["tokens"] = tokens
        sample["mask"] = mask
        sample["images"] = images
        return sample

    @property
    def eos_id(self):
        return -1

    @property
    def bos_id(self):
        return 0

    @property
    def image_id(self):
        return -2


class DummyChatFormat:

    B_SYS, E_SYS = "System:\n", "\n"
    B_INST, E_INST = "User:\n", "\nAssistant:\n"
    B_ASST, E_ASST = "", ""
    system = f"{B_SYS}{{content}}{E_SYS}"
    user = f"{B_INST}{{content}}{E_INST}"
    assistant = f"{B_ASST}{{content}}{E_ASST}"

    @classmethod
    def format(
        cls,
        messages,
    ):
        formats = {"system": cls.system, "user": cls.user, "assistant": cls.assistant}
        formatted_dialogue = []
        for message in messages:
            content = formats.get(message.role).format(
                content=message.content[0]["content"]
            )
            formatted_dialogue.append(
                Message(role=message.role, content=content, masked=message.masked),
            )
        return formatted_dialogue


DummyPromptTemplate = partial(
    PromptTemplate,
    template={
        "system": ("System:\n", "\n"),
        "user": ("User:\n", "\n"),
        "assistant": ("Assistant:\n", "\n"),
    },
)


def get_assets_path():
    return Path(__file__).parent / "assets"


def fixed_init_tensor(
    shape: torch.Size,
    min_val: Union[float, int] = 0.0,
    max_val: Union[float, int] = 1.0,
    nonlinear: bool = False,
    dtype: torch.dtype = torch.float,
):
    """
    Utility for generating deterministic tensors of a given shape. In general stuff
    like torch.ones, torch.eye, etc can result in trivial outputs. This utility
    generates a range tensor [min_val, max_val) of a specified dtype, applies
    a sine function if nonlinear=True, then reshapes to the appropriate shape.
    """
    n_elements = math.prod(shape)
    step_size = (max_val - min_val) / n_elements
    x = torch.arange(min_val, max_val, step_size, dtype=dtype)
    x = x.reshape(shape)
    if nonlinear:
        return torch.sin(x)
    return x


@torch.no_grad
def fixed_init_model(
    model: nn.Module,
    min_val: Union[float, int] = 0.0,
    max_val: Union[float, int] = 1.0,
    nonlinear: bool = False,
    dtype: Optional[torch.dtype] = None,
):
    """
    This utility initializes all parameters of a model deterministically using the
    function fixed_init_tensor above. See that docstring for details of each parameter.
    """
    for _, param in model.named_parameters():
        param.copy_(
            fixed_init_tensor(
                param.shape,
                min_val=min_val,
                max_val=max_val,
                nonlinear=nonlinear,
                dtype=param.dtype if dtype is None else dtype,
            )
        )


def assert_expected(
    actual: Any,
    expected: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_device: bool = True,
):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        check_device=check_device,
        msg=f"actual: {actual}, expected: {expected}",
    )


@contextmanager
def single_box_init(init_pg: bool = True):
    env_vars = ["MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK", "RANK", "WORLD_SIZE"]
    initial_os = {k: os.environ.get(k, None) for k in env_vars}
    os.environ.get("MASTER_ADDR", None)
    os.environ["MASTER_ADDR"] = "localhost"
    # TODO: Don't hardcode ports as this could cause flakiness if tests execute
    # in parallel.
    os.environ["MASTER_PORT"] = str(12345)
    os.environ["LOCAL_RANK"] = str(0)
    os.environ["RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(1)
    if init_pg:
        torch.distributed.init_process_group(
            backend="gloo",
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"]),
        )
    try:
        yield
    finally:
        if init_pg:
            torch.distributed.destroy_process_group()
        for k in env_vars:
            if initial_os.get(k) is None:
                del os.environ[k]
            else:
                os.environ[k] = initial_os[k]


@contextmanager
def set_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


@contextmanager
def captured_output() -> Generator[Tuple[TextIO, TextIO], None, None]:
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def gpu_test(gpu_count: int = 1):
    """
    Annotation for GPU tests, skipping the test if the
    required amount of GPU is not available
    """
    message = f"Not enough GPUs to run the test: requires {gpu_count}"
    local_gpu_count: int = torch.cuda.device_count()
    return pytest.mark.skipif(local_gpu_count < gpu_count, reason=message)


def get_loss_values_from_metric_logger(log_file_path: str) -> Dict[str, float]:
    """
    Given an output directory containing metric logger .txt file,
    parse the .txt and return a list of losses from each logged iteration.
    """
    with open(log_file_path, "r") as f:
        logs = f.read()
    losses = [float(x) for x in re.findall(r"loss:(\d+\.\d+)", logs)]
    return losses


def gen_log_file_name(tmpdir, suffix: Optional[str] = None) -> str:
    """
    Take the tmpdir and just append a non-path version of it as the
    filename, optionally adding specified suffix. This is used to
    write metric logs to a deterministic file per test run.
    E.g. /tmp/my/dir -> /tmp/my/dir/tmpmydir.txt
    """
    filename = str(tmpdir) + str(tmpdir).replace("/", "")
    if suffix:
        filename += suffix
    filename += ".txt"
    return filename


def assert_dialogue_equal(actual, expected):
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        assert actual[i].role == expected[i].role
        assert actual[i].text_content == expected[i].text_content


def mps_ignored_test() -> bool:
    return pytest.mark.skipif(
        torch.backends.mps.is_available() and torch.backends.mps.is_built(),
        reason="Test skipped due to torch being compiled with MPS"
        "see https://github.com/pytorch/torchtune/issues/1707 for more information",
    )
