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
        [ { "from": "system", "value": "You are an AI assistant. You will be given a task. You must generate a detailed and long answer." }, { "from": "human", "value": "Predecesorul său, primul îngrijitor al moscheii Et'hem Bey, a murit în timp ce era în moschee; sute de persoane au participat la funeraliile sale.\n\nWhich language is this?" }, { "from": "gpt", "value": "This text is written in Romanian. The passage discusses the predecessor of someone who was the first caretaker of the Et'hem Bey Mosque and mentions that they passed away while in the mosque. It also notes that hundreds of people attended their funeral." } ]

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
        # from : system (prompt), human (instruction), gpt (response)
        for conversation in data:
            agent = conversation["from"]
            text = conversation["value"]
            agent_text_dict[agent] = text
        return self.transform(
            agent_text_dict["system"], agent_text_dict["human"], agent_text_dict["gpt"]
        )

    def transform(
        self, prompt: str, instruction: str, response: str
    ) -> Tuple[List[int], List[int]]:
        # Add instruction and response tags to construct the input string
        instruction_tag = "\n\n### Instruction:\n"
        response_tag = "\n\n### Response:\n"
        instructions_and_inputs = self._tokenizer.encode(
            prompt + instruction_tag + instruction + response_tag
        )
        labels = self._tokenizer.encode(response)
        return instructions_and_inputs, labels
