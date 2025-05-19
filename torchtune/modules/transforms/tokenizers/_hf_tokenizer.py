# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import jinja2
from jinja2 import StrictUndefined

from tokenizers import Tokenizer
from torchtune.data import Message, truncate
from torchtune.modules.transforms.tokenizers._utils import BaseTokenizer, ModelTokenizer


class HuggingFaceBaseTokenizer(BaseTokenizer):
    """
    A wrapper around Hugging Face tokenizers. See https://github.com/huggingface/tokenizers
    This can be used to load from a Hugging Face tokenizer.json file into a torchtune BaseTokenizer.

    This class will load the tokenizer.json file from tokenizer_json_path. It will
    attempt to infer BOS and EOS token IDs from config.json if possible, and if not
    will fallback to inferring them from generation_config.json.

    Args:
        tokenizer_json_path (str): Path to tokenizer.json file
        tokenizer_config_json_path (Optional[str]): Path to tokenizer_config.json file. Default: None
        generation_config_path (Optional[str]): Path to generation_config.json file.
            Default: None

    Raises:
        ValueError: If neither tokenizer_config_json_path or generation_config_path are specified.
    """

    def __init__(
        self,
        tokenizer_json_path: str,
        *,
        tokenizer_config_json_path: Optional[str] = None,
        generation_config_path: Optional[str] = None,
    ):
        self.tokenizer = Tokenizer.from_file(tokenizer_json_path)
        if not (tokenizer_config_json_path or generation_config_path):
            raise ValueError(
                "At least one of tokenizer_config_json_path or generation_config_path must be specified."
            )
        if tokenizer_config_json_path:
            with open(tokenizer_config_json_path, "rb") as f:
                self.config = json.load(f)
        else:
            self.config = None
        if generation_config_path:
            with open(generation_config_path, "rb") as f:
                self.generation_config = json.load(f)
        else:
            self.generation_config = None
        self._infer_bos_eos_tokens()
        self._infer_should_add_bos_eos()

    def _get_token_from_config(self, config: Dict[str, Any], key: str) -> str:
        """
        HF BOS/EOS tokens are either stored as e.g. {'bos_token': 5}
        or {'bos_token': {'content': 5, ...}}. This utility handles both.
        """
        token = config.get(key)
        if isinstance(token, Dict):
            if "content" not in token:
                raise ValueError(f"Could not parse {key} from config")
            token = token["content"]
        else:
            if not isinstance(token, str):
                raise ValueError(f"Could not parse {key} from config")
        return token

    def _infer_bos_eos_tokens(self):
        """
        Infer BOS and EOS token IDs from config and/or generation_config.

        Will first try to infer token from config then map to ID.
        If that's not available, will infer ID directly from generation_config.
        Otherwise, raise a ValueError.
        """
        self.bos_id = None
        self.eos_id = None

        if self.config:
            bos_token = self._get_token_from_config(self.config, "bos_token")
            eos_token = self._get_token_from_config(self.config, "eos_token")
            if bos_token is not None:
                self.bos_id = self.tokenizer.token_to_id(bos_token)
            if eos_token is not None:
                self.eos_id = self.tokenizer.token_to_id(eos_token)

        if self.generation_config:
            if self.bos_id is None:
                self.bos_id = self.generation_config.get("bos_token_id")
            if self.eos_id is None:
                self.eos_id = self.generation_config.get("eos_token_id")

        if self.bos_id is None or self.eos_id is None:
            raise ValueError("Could not infer BOS and EOS token IDs from config")

    def _infer_should_add_bos_eos(self):
        """
        Hugging Face tokenizers sometimes add BOS by default. We should infer this to determine
        whether to add it ourselves in encode. Otherwise we will get duplicate BOS tokens.
        """

        self.hf_adds_bos, self.hf_adds_eos = False, False
        encoded_empty_str = self.tokenizer.encode("").ids

        if self.bos_id in encoded_empty_str:
            self.hf_adds_bos = True
        if self.eos_id in encoded_empty_str:
            self.hf_adds_eos = True

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> List[int]:
        """
        Encodes a string into a list of token ids.

        Args:
            text (str): The text to encode.
            add_bos (bool): Whether to add the tokenizer's bos_id to the encoded string.
                Default True.
            add_eos (bool): Whether to add the tokenizer's eos_id to the encoded string.
                Default True.

        Returns:
            List[int]: The list of token ids.
        """
        token_ids = self.tokenizer.encode(text).ids
        if add_bos and not self.hf_adds_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos and not self.hf_adds_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token ids into a string.

        Args:
            token_ids (List[int]): The list of token ids.

        Returns:
            str: The decoded string.
        """
        return self.tokenizer.decode(token_ids)


def _infer_special_tokens_from_hf_config(config: dict) -> List[str]:
    special_tokens = set()

    standard_keys = [
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
        "sep_token",
        "cls_token",
        "mask_token",
    ]

    for key in standard_keys:
        if token := config.get(key):
            if isinstance(token, str):
                content = token
            else:
                content = token.get("content")

            if content:
                special_tokens.add(content)

    for token in config.get("additional_special_tokens", []):
        if isinstance(token, str):
            content = token
        else:
            content = token.get("content")

        if content:
            special_tokens.add(content)

    for token_info in config.get("added_tokens_decoder", {}).values():
        if token_info.get("special", False):
            if content := token_info.get("content"):
                special_tokens.add(content)

    return list(special_tokens)


class HuggingFaceModelTokenizer(ModelTokenizer):
    def __init__(
        self,
        tokenizer_json_path: str,
        *,
        tokenizer_config_json_path: Optional[str] = None,
        generation_config_path: Optional[str] = None,
        truncation_type: str = "right",
    ):
        self.base_tokenizer = HuggingFaceBaseTokenizer(
            tokenizer_json_path=tokenizer_json_path,
            tokenizer_config_json_path=tokenizer_config_json_path,
            generation_config_path=generation_config_path,
        )

        # Contents of the tokenizer_config.json
        config = self.base_tokenizer.config

        self.special_tokens = _infer_special_tokens_from_hf_config(config)
        self.top_level_variables = self.extract_top_level_variables(config)

        _env = jinja2.Environment(undefined=StrictUndefined)

        # It is used sometimes in HF chat_templates
        _env.globals["raise_exception"] = self._raise_helper

        self.template = _env.from_string(
           config["chat_template"]
        )
        self.truncation_type = truncation_type

        self.special_tokens_mapping = {}
        for token in self.special_tokens:
            self.special_tokens_mapping[token] = self.base_tokenizer.encode(token)

    def _raise_helper(self, message: str):
        raise jinja2.exceptions.TemplateError(message)

    def extract_top_level_variables(self, config):
        top_level = {}
        for key, value in config.items():
            if not isinstance(value, (dict, list)):
                top_level[key] = value
        return top_level

    def tokenize_messages(
        self,
        messages: List[Message],
        add_eos: bool = True,
        max_seq_len: Union[int, None] = None,
    ) -> Tuple[List[int], List[bool]]:
        tokenized_messages = []
        mask = []
        previous_tokens = []

        for i, message in enumerate(messages):
            current_messages = [
                {"role": m.role, "content": m.content[0]["content"]}
                for m in messages[: i + 1]
            ]

            rendered = self.template.render(
                messages=current_messages,
                add_generation_prompt=add_eos if i == len(messages) - 1 else False,
                **self.special_tokens_mapping,  # We assume that the naming is consistent
                **self.top_level_variables,
            )

            current_tokens = self.base_tokenizer.encode(rendered, add_eos=False)

            delta = current_tokens[len(previous_tokens) :]
            previous_tokens = current_tokens
            tokenized_messages.extend(delta)

            mask.extend([message.masked] * len(delta))

        if add_eos and self.base_tokenizer.eos_id is not None:
            tokenized_messages.append(self.base_tokenizer.eos_id)
            mask.append(False)

        # Finally, truncate if necessary
        tokenized_messages = truncate(
            tokens=tokenized_messages,
            max_seq_len=max_seq_len,
            eos_id=self.base_tokenizer.eos_id,
            truncation_type=self.truncation_type,
        )

        mask = truncate(
            tokens=mask,
            max_seq_len=max_seq_len,
            eos_id=True if add_eos else None,
            truncation_type=self.truncation_type,
        )

        return tokenized_messages, mask
