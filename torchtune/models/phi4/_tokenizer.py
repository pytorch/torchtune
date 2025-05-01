# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Mapping, Optional, Tuple

from torchtune.data import Message, PromptTemplate, truncate
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import GPT2BaseTokenizer

PHI4_SPECIAL_TOKENS = {
    "<|dummy_0|>": 100256,
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|dummy_1|>": 100261,
    "<|dummy_2|>": 100262,
    "<|dummy_3|>": 100263,
    "<|im_start|>": 100264,
    "<|im_end|>": 100265,
    "<|im_sep|>": 100266,
    "<|endofprompt|>": 100276,
}

# Add all <dummy_x>
current_dummy_index = 4
for token_id in range(100267, 100352):
    if token_id == 100276:
        continue  # Skip the token_id that's already assigned to <|endofprompt|>
    PHI4_SPECIAL_TOKENS[f"<|dummy_{current_dummy_index}|>"] = token_id
    current_dummy_index += 1

CL100K_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa


class Phi4Tokenizer(ModelTokenizer, Transform):
    """
    TikToken tokenizer configured with Phi4 (14B) special tokens.

    Args:
        merges_path (str): Path to merges.txt file.
        vocab_path (str): Path to vocab.json file.
        special_tokens (Optional[Dict[str, int]]): Mapping containing special text tokens and
            their registered token IDs. If left as None, this will be set to the canonical
            Phi4 special tokens.
        max_seq_len (Optional[int]): Max sequence length to truncate tokens to.
        prompt_template (Optional[PromptTemplate]): Template used to format the messages based on their role.
        truncation_type (str): type of truncation to apply, either "left" or "right".
            Default is "right".
    """

    def __init__(
        self,
        merges_path: str = None,
        vocab_path: str = None,
        special_tokens: Optional[Dict[str, int]] = None,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[PromptTemplate] = None,
        truncation_type: str = "right",
    ):
        self.special_tokens = special_tokens or PHI4_SPECIAL_TOKENS

        # Use custom EOS, BOS, and pad ids instead of GPT2
        self.eos_id = self.special_tokens["<|im_end|>"]
        self.bos_id = self.special_tokens["<|endoftext|>"]
        self.pad_id = self.special_tokens["<|dummy_85|>"]

        self.stop_tokens = [self.eos_id]
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template

        self.tokenizer_model = GPT2BaseTokenizer(
            vocab_path,
            merges_path,
            self.eos_id,
            self.bos_id,
            self.eos_id,
            self.pad_id,
        )

        self.truncation_type = truncation_type

    @property
    def vocab_size(self):
        return self.tokenizer_model.vocab_size

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> List[int]:
        return self.tokenizer_model.encode(text=text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to strings."""
        ids_for_decode = [
            token_id
            for token_id in ids
            if not (skip_special_tokens and 100_256 <= token_id <= 100_351)
        ]
        return self.tokenizer_model.decode(ids_for_decode)

    def _tokenize_header(self, role: str) -> list:
        tokenized_messages = [self.special_tokens["<|im_start|>"]]
        tokenized_messages.extend(self.encode(role, add_bos=False, add_eos=False))
        tokenized_messages.append(self.special_tokens["<|im_sep|>"])
        return tokenized_messages

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_eos: bool = False,
        ignore_system_prompt: bool = False,
    ) -> Tuple[List[int], List[bool]]:
        templated_messages = (
            self.prompt_template(messages) if self.prompt_template else messages
        )

        tokenized_messages = []
        mask = []

        for message in templated_messages:
            if ignore_system_prompt and message.role == "system":
                continue

            tokenized_header = self._tokenize_header(message.role)
            tokenized_messages.extend(tokenized_header)
            mask.extend([message.masked] * len(tokenized_header))

            tokens = []
            for item in message.content:
                if item["type"] == "text":
                    tokens += self.encode(
                        item["content"].rstrip(" "), add_bos=False, add_eos=False
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported message content type: {item['type']}"
                    )

            if add_eos and message.role == "assistant":
                tokens.append(self.special_tokens["<|im_end|>"])
            elif message.role != "assistant":
                tokens.append(self.special_tokens["<|im_end|>"])

            tokenized_messages.extend(tokens)
            mask.extend([message.masked] * len(tokens))

            if self.max_seq_len and len(tokenized_messages) >= self.max_seq_len:
                break

        # Finnaly, truncate if necessary.
        if self.max_seq_len and len(tokenized_messages) >= self.max_seq_len:
            tokenized_messages = truncate(
                tokens=tokenized_messages,
                max_seq_len=self.max_seq_len,
                eos_id=self.eos_id if add_eos else None,
                truncation_type=self.truncation_type,
            )
            mask = truncate(
                tokens=mask,
                max_seq_len=self.max_seq_len,
                eos_id=True if add_eos else None,
                truncation_type=self.truncation_type,
            )

        return tokenized_messages, mask

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Apply `tokenize_messages` to the "messages" field in the sample.
        """
        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample
