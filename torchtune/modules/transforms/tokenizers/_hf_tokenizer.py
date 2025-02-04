import json
from typing import List

from tokenizers import Tokenizer
from torchtune.modules.transforms.tokenizers._utils import BaseTokenizer


class HFTokenizer(BaseTokenizer):
    """
    A wrapper around HuggingFace tokenizers. BLAH BLAH BLAH

    Args:
        path (str): Path to tokenizer.json file
        config_path (str): Path to tokenizer_config.json file
    """

    def __init__(self, path: str, config_path: str):
        self.hf_tokenizer = Tokenizer.from_file(path)
        with open(config_path, "rb") as f:
            config = json.load(f)

    def _infer_tokenizer_class_from_config(self):
        pass

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        """
        Encodes a string into a list of token ids.

        Args:
            text (str): The text to encode.
            add_bos (bool): Whether to add a beginning-of-sequence token to the beginning of the
        """
        token_ids = self.hf_tokenizer.encode(text).ids
        if add_bos:
            token_ids.insert(0, self.hf_tokenizer.bos_id)
        if add_eos:
            token_ids.append(self.hf_tokenizer.eos_id)
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a list of token ids into a string.

        Args:
            token_ids (List[int]): The list of token ids to decode.
        """
        return self.hf_tokenizer.decode(token_ids)
