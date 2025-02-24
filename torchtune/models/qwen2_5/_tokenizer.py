# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Optional, Tuple

from torchtune.data import ChatMLTemplate, Message, PromptTemplate, truncate
from torchtune.models.qwen2._tokenizer import (
    DEFAULT_QWEN2_TOKENIZER_BPE_CACHE_SIZE,
    ENDOFTEXT,
    IM_END,
    QWEN2_SPECIAL_TOKENS,
    Qwen2Tokenizer,
)


QWEN2_5_SPECIAL_TOKENS = {
    **QWEN2_SPECIAL_TOKENS,
    "<|object_ref_start|>": 151646,
    "<|object_ref_end|>": 151647,
    "<|box_start|>": 151648,
    "<|box_end|>": 151649,
    "<|quad_start|>": 151650,
    "<|quad_end|>": 151651,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|vision_pad|>": 151654,
    "<|image_pad|>": 151655,
    "<|video_pad|>": 151656,
    "<tool_call>": 151657,
    "</tool_call>": 151658,
    "<|fim_prefix|>": 151659,
    "<|fim_middle|>": 151660,
    "<|fim_suffix|>": 151661,
    "<|fim_pad|>": 151662,
    "<|repo_name|>": 151663,
    "<|file_sep|>": 151664,
}


class Qwen2_5Tokenizer(Qwen2Tokenizer):  # noqa: N801
    """This class construct a Qwen2.5 tokenizer, based on GPT-2 byte-level BPE tokenization.

    See <https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/qwen2/tokenization_qwen2.py>
    and <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json>.

    Args:
        path (str): Path to vocab.json file.
        merges_file (str): Path to merges.txt file.
            merges.txt contains all BPE merge operations, and this file is required to split a single word into
            byte-level BPE tokens.
        special_tokens (Dict[str, int]): Special tokens to add to the tokenizer. Default is QWEN2_5_SPECIAL_TOKENS.
        max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
            Default: None
        prompt_template (Optional[PromptTemplate]): template used to format the messages based on their role. This is used
            to add structured text around the actual messages. The structured text is used in three scenarios:

            - Task-specific templates to gear models for a particular task that it will expect after training
            - Model-specific templates that are required whenever the model is prompted, such as the [INST]
              tags in Llama2 and in Mistral
            - Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

            The extra text will still get tokenized as normal text, not as special tokens.
            Default: None
        errors (str): Paradigm to follow when decoding bytes to UTF-8. Defaults to "replace".
            See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (Optional[str]): The unknown token. A token that is not in the vocabulary cannot be converted
            to an ID and is set to be this token instead. Defaults to ``<|endoftext|>``.
        bos_token (Optional[str]): The beginning of sequence token. Defaults to None.
        eos_token (str): The end of sequence token. Defaults to ``<|endoftext|>``.
        pad_token (Optional[str]): The token used for padding. Defaults to ``<|endoftext|>``.
        bpe_cache_size (int): BPE token cache size in Qwen2Tokenizer.
            NOTE: large cache size will speed up tokenization, but the cache object will get really
            large for long running processes (esp. for texts of language that do not use space between
            word, e.g. Chinese); technically not a memory leak but appears as one.
            By default, we set the cache size equals to size of the official Qwen2 tokenizer.
        truncation_type (str): type of truncation to apply, either "left" or "right".
            Default is "right".

    Example:
        >>> tokenizer = Qwen2Tokenizer(
                path="/path/to/vocab.json", merges_file="/path/to/merges.txt", special_tokens=QWEN2_SPECIAL_TOKENS)
        >>> tokenized_text = tokenizer.encode("Hello world!")
        >>> print(tokenized_text)
        [39, 385, 78, 675, 0, 2000]
    """

    def __init__(
        self,
        path: str,
        merges_file: str,
        special_tokens: Dict[str, int] = QWEN2_5_SPECIAL_TOKENS,
        max_seq_len: Optional[int] = None,
        *,
        prompt_template: Optional[PromptTemplate] = None,
        errors: str = "replace",
        unk_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: str = IM_END,
        pad_token: Optional[str] = ENDOFTEXT,
        bpe_cache_size: int = DEFAULT_QWEN2_TOKENIZER_BPE_CACHE_SIZE,
        truncation_type: str = "right",
    ):
        super().__init__(
            path=path,
            merges_file=merges_file,
            special_tokens=special_tokens,
            max_seq_len=max_seq_len,
            prompt_template=prompt_template,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            bpe_cache_size=bpe_cache_size,
        )

        self.tool_call_start_id = self.special_tokens["<tool_call>"]
        self.tool_call_end_id = self.special_tokens["</tool_call>"]
        self.truncation_type = truncation_type

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_eos: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Given a list of messages, return a list of tokens for the concatenated
        and formatted messages.

        Args:
            messages (List[Message]): The message list to tokenize.
            add_eos (bool): Wether to add the tokenizer's eos_id at the end of the
                sequence of messages. Default is True.

        Returns:
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.

        Raises:
            RuntimeError: If a message contains non-text content
        """
        assert not isinstance(self.prompt_template, ChatMLTemplate), (
            "Using ChatMLTemplate with tokenize_messages will result in multiple <|im_*|> tokens wrapping each message."
            "Please use a different template or set to None."
        )
        templated_messages = (
            self.prompt_template(messages)
            if self.prompt_template is not None
            else messages
        )

        tokenized_messages = []
        mask = []
        for i, message in enumerate(templated_messages):
            # message header
            tokens = self._tokenize_header(templated_messages, i)

            # message content
            for item in message.content:
                if item["type"] == "text":
                    tokens.extend(
                        self.encode(
                            item["content"],
                            add_bos=False,
                            add_eos=False,
                        )
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported message content type: {item['type']}"
                    )

            # message footer
            tokens.extend(self._tokenize_footer(templated_messages, i))

            tokenized_messages.extend(tokens)
            mask.extend([message.masked] * len(tokens))

            # Break out early if we reach max_seq_len
            if self.max_seq_len and len(tokenized_messages) >= self.max_seq_len:
                break

        # Add the End-Of-Sequence token
        if add_eos:
            tokenized_messages.append(self.eos_id)
            mask.append(mask[-1])

        # Finally, truncate if necessary
        if self.max_seq_len:
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

    def _tokenize_header(self, messages, i):
        tokens = []
        message = messages[i]
        if message.role == "ipython":
            if i == 0 or messages[i - 1].role != "ipython":
                # only add the "user" header if this is the first tool response msg
                self._add_message_start_tokens(tokens, "user")
                tokens.extend(
                    self.encode("<tool_response>\n", add_bos=False, add_eos=False)
                )
            else:
                tokens.extend(
                    self.encode("\n<tool_response>\n", add_bos=False, add_eos=False)
                )
        else:
            self._add_message_start_tokens(tokens, message.role)
            if message.role == "assistant" and message.ipython:
                tokens.append(self.tool_call_start_id)
                tokens.extend(self.encode("\n", add_bos=False, add_eos=False))
        return tokens

    def _tokenize_footer(self, messages, i):
        tokens = []
        message = messages[i]
        if message.role == "ipython":
            if i == len(messages) - 1 or messages[i + 1].role != "ipython":
                tokens.extend(
                    self.encode("\n</tool_response>", add_bos=False, add_eos=False)
                )
                self._add_message_end_tokens(tokens)
            else:
                tokens.extend(
                    self.encode("\n</tool_response>", add_bos=False, add_eos=False)
                )
        else:
            if message.role == "assistant" and message.ipython:
                tokens.extend(self.encode("\n", add_bos=False, add_eos=False))
                tokens.append(self.tool_call_end_id)
            if message.role != "assistant" or i != len(messages) - 1:
                self._add_message_end_tokens(tokens)
        return tokens

    def _add_message_start_tokens(self, tokens, role):
        tokens.append(self.im_start_id)
        tokens.extend(self.encode(f"{role}\n", add_bos=False, add_eos=False))

    def _add_message_end_tokens(self, tokens):
        tokens.append(self.im_end_id)
        tokens.extend(self.encode("\n", add_bos=False, add_eos=False))
