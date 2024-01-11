# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

# Not ideal to import this type here but it's needed for the transform function
from torchtune.models.llama2.tokenizer import Tokenizer


class SlimOrcaDataset(Dataset):
    """PyTorch Representation of the SlimOrca Dataset from Hugging Face.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.

    Data input format:
        [ { "from": "system", "value": "You are an AI assistant. You will be
        given a task. You must generate a detailed and long answer." },
        { "from": "human", "value": "Predecesorul său, primul îngrijitor al
        moscheii Et'hem Bey, a murit în timp ce era în moschee; sute de
        persoane au participat la funeraliile sale.\n\nWhich language is this?
        " }, { "from": "gpt", "value": "This text is written in Romanian.
        The passage discusses the predecessor of someone who was the first
        caretaker of the Et'hem Bey Mosque and mentions that they passed away
        while in the mosque. It also notes that hundreds of people attended
        their funeral." } ]

    Example:
    >>> slimorca_ds = SlimOrcaDataset(tokenizer=tokenizer)
    >>> for batch in Dataloader(slimorca_ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """

    def __init__(self, tokenizer: Tokenizer, **kwargs) -> None:
        self._data = load_dataset("Open-Orca/SlimOrca-Dedup", split="train")
        self._tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        data = self._data[index]["conversations"]
        agent_text_dict = {}
        # agents can be {system, human, gpt}
        for conversation in data:
            agent = conversation["from"]
            text = conversation["value"]
            agent_text_dict[agent] = text

        # If system value is present
        if len(agent_text_dict["system"]) > 0:
            prompt = f"<s>[INST] <<SYS>> {agent_text_dict['system']} <</SYS>> {agent_text_dict['human']} [/INST]"
        else:
            prompt = f"<s>[INST] {agent_text_dict['human']} [/INST]"

        # prompt_and_response = prompt +
        input_tokens = self._tokenizer(prompt_and_response)
        reponse_tokens = self._tokenizer(f" {agent_text_dict['gpt']} </s>")
        length_input_tokens = len(input_tokens)

        labe = [_CROSS_ENTROPY_IGNORE_IDX for _ in ramge(len(As))]

        return self._tokenizer.encode(prompt), self._tokenizer.encode(
            f"{agent_text_dict['gpt']} </s>"
        )
