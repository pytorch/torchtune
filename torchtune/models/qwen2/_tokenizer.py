# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, List, Tuple

from tokenizers import Tokenizer as TokenizerFast

from torchtune.data import Message, truncate
from torchtune.modules.tokenizers import ModelTokenizer


ENDOFTEXT = "<|endoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


class Qwen2Tokenizer(ModelTokenizer):
    """This class construct a "fast" Qwen2 tokenizer (backed by HuggingFace's *tokenizers* library).

    See <https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/qwen2/tokenization_qwen2_fast.py>.

    Args:
        path (str): Path to tokenizer.json file.

    Example:
        >>> tokenizer = Qwen2Tokenizer("/path/to/tokenizer.json")
        >>> tokenized_text = tokenizer.encode("Hello world!")
        >>> print(tokenized_text)
        []
    """

    system = f"{IM_START}system\n{{content}}{IM_END}\n"
    user = f"{IM_START}user\n{{content}}{IM_END}\n"
    assistant = f"{IM_START}assistant\n{{content}}{IM_END}\n"
    assistant_for_generation = f"{IM_START}assistant\n"

    def __init__(
        self,
        path: str,
        *,
        unk_token: Optional[str] = ENDOFTEXT,
        bos_token: Optional[str] = None,
        eos_token: str = ENDOFTEXT,
        pad_token: Optional[str] = ENDOFTEXT,
    ):
        # Build backend tokenizer.
        self._tokenizer = TokenizerFast.from_file(path)

        _truncation = self._tokenizer.truncation
        if _truncation is not None:
            self._tokenizer.enable_truncation(**_truncation)
        else:
            self._tokenizer.no_truncation()

        _padding = self._tokenizer.padding
        if _padding is not None:
            self._tokenizer.enable_padding(**_padding)

        vocab = self._tokenizer.get_vocab()
        self.unk_id = None if unk_token is None else vocab[unk_token]
        self.bos_id = None if bos_token is None else vocab[bos_token]
        self.eos_id = None if eos_token is None else vocab[eos_token]
        self.pad_id = None if pad_token is None else vocab[pad_token]
        self.im_start_id = vocab[IM_START]
        self.im_end_id = vocab[IM_END]
        self.stop_tokens = [self.eos_id, self.im_end_id]

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = True, **kwargs
    ) -> List[int]:
        """
        Encode a string into a list of token ids.

        Args:
            text (str): The string to encode.
            add_bos (bool): (Optional) Whether to add the beginning of sequence token.
            add_eos (bool): (Optional) Whether to add the end of sequence token.

        Returns:
            List[int]: The list of token ids.
        """
        return self.encode_batch([text], add_bos=add_bos, add_eos=add_eos, **kwargs)[0]

    def encode_batch(
        self,
        batch_text: List[str],
        add_bos: bool = True,
        add_eos: bool = True,
        **kwargs,
    ) -> List[List[int]]:
        """Encode a batch of strings into lists of token ids.

        Args:
            batch_text (List[str]): The batch of strings to encode.
            add_bos (bool): (Optional) Whether to add the beginning of sequence token.
            add_eos (bool): (Optional) Whether to add the end of sequence token.

        Returns:
            List[List[int]]: A batch of lists of token ids.
        """
        encodings = self._tokenizer.encode_batch(batch_text)
        encoded_token_ids = []
        for encoding in encodings:
            encoding_ids = encoding.ids[:]
            if add_bos and self.bos_id is not None:
                encoding_ids.insert(0, self.bos_id)
            if add_eos and self.eos_id is not None:
                encoding_ids.append(self.eos_id)
            encoded_token_ids.append(encoding_ids)
        return encoded_token_ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        """
        Decode a list of token ids into a string.

        Args:
            token_ids (List[int]): The list of token ids.
            skip_special_tokens (bool): Whether the special tokens should be removed from the decoded string.

        Returns:
            str: The decoded string.
        """
        text = self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        return text

    def tokenize_messages(
        self,
        messages: List[Message],
        max_seq_len: Optional[int] = None,
        apply_chat_template: bool = True,
        **kwargs,
    ) -> Tuple[List[int], List[bool]]:
        """
        Given a list of messages, return a list of tokens for the concatenated
        and formatted messages.

        Args:
            messages (List[Message]): The message list to tokenize.
            max_seq_len (Optional[int]): The maximum sequence length.
            apply_chat_template (bool): Whether to apply Qwen2 chat template.

        Returns:
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.
        """
        tokens = []
        mask = []
        is_generation = False
        for index, message in enumerate(messages):
            content = ""
            if message.role == "system":
                content = self.system.format(content=message.content)
            elif message.role == "user":
                content = self.user.format(content=message.content)
            elif message.role == "assistant":
                if index == len(messages) - 1 and not message.content:
                    content = self.assistant_for_generation
                    is_generation = True
                else:
                    content = self.assistant.format(content=message.content)
            tokenized_message = self.encode(content, add_bos=False, add_eos=False)
            tokens.extend(tokenized_message)
            mask.extend([message.masked] * len(tokenized_message))

            if max_seq_len and len(tokens) >= max_seq_len:
                break

        if not is_generation:
            tokens = tokens + [self.eos_id]
            last_message_masked = False
            if messages:
                last_message_masked = messages[-1].masked
            mask = mask + [last_message_masked]
        if max_seq_len:
            tokens = truncate(tokens, max_seq_len, self.eos_id)
            mask = truncate(mask, max_seq_len, True)
        return tokens, mask
