# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

# Not ideal to import this type here but it's needed for the transform function
from torchtune.modules import Tokenizer

_CROSS_ENTROPY_IGNORE_IDX = -100


class SlimOrcaDataset(Dataset):
    """PyTorch Representation of the SlimOrca Dataset
    https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup
    from Hugging Face.

    The data is formatted to adhere to Llama2 Chat Format.
    This format is required if the base model is Llama2 Chat Model.

    The returned data is a tuple of input token id list and label token id
    list. If `max_token_length` keyword argument is provided, the returned
    input token id list is ensured (by truncation if necssary) to be within
    that length.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        **kwargs: Additional keyword arguments to pass to the SlimOrca Dataset.
        This value needs to be at least 4 though it is generally set it to
        max sequence length accepted by the model.

    Keyword Arguments:
        max_token_length (int): Maximum number of tokens in the returned.
        Default is 1024.

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

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def __init__(self, tokenizer: Tokenizer, **kwargs) -> None:
        self._data = load_dataset("Open-Orca/SlimOrca-Dedup", split="train")
        self._tokenizer = tokenizer
        self._max_token_length = kwargs.get("max_token_length", 1024)
        if self._max_token_length < 4:
            # Input token needs to have 1 bos, 1 eos,
            # and 1 token from prompt, 1 from label
            raise ValueError("max_token_length must be at least 4")

    def __len__(self):
        return len(self._data)

    def prompt_with_system(self, content: str) -> str:
        return f"{self.B_INST} {self.B_SYS}{content}{self.E_SYS} {self.E_INST}"

    def prompt_without_system(self, content: str) -> str:
        return f"{self.B_INST} {content} {self.E_INST}"

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        data = self._data[index]["conversations"]
        prompt, label = self.generate_prompt_label(data)
        return self.generate_tokens(prompt, label)

    def generate_tokens(self, prompt: str, label: str) -> Tuple[List[int], List[int]]:
        prompt_tokens = self._tokenizer.encode(prompt, add_bos=True, add_eos=False)
        # Truncate to max token length - 2 (so that there is at least one label token)
        prompt_tokens = prompt_tokens[: self._max_token_length - 2]

        # Calculate space left for label tokens
        label_tokens_length = self._max_token_length - len(prompt_tokens)
        label_tokens = self._tokenizer.encode(label, add_bos=False, add_eos=True)

        # Truncate label tokens
        label_tokens = label_tokens[: label_tokens_length - 1]
        if label_tokens[-1] != self._tokenizer.eos_id:
            label_tokens.append(self._tokenizer.eos_id)

        input = prompt_tokens + label_tokens
        label = [
            _CROSS_ENTROPY_IGNORE_IDX for _ in range(len(prompt_tokens))
        ] + label_tokens
        assert len(input) == len(label)
        return input, label

    def generate_prompt_label(self, data: List[Dict[str, str]]) -> Tuple[str, str]:
        agent_text_dict = {}
        # agents can be {system, human, gpt}
        for conversation in data:
            agent = conversation["from"]
            text = conversation["value"]
            agent_text_dict[agent] = text

        # Llama2 Chat Format - https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L284
        if len(agent_text_dict["system"]) > 0:
            prompt = f"{self.B_INST} {self.B_SYS}{agent_text_dict['system']}{self.E_SYS}{agent_text_dict['human']} {self.E_INST}"
        else:
            prompt = f"{self.B_INST} {agent_text_dict['human']} {self.E_INST}"

        response = f" {agent_text_dict['gpt']} "
        return prompt, response
