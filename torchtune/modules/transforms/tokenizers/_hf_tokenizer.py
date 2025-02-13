# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Dict, List, Optional

from tokenizers import Tokenizer
from torchtune.modules.transforms.tokenizers._utils import BaseTokenizer


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
