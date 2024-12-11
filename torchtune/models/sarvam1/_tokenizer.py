# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping, Optional, Tuple

from torchtune.data import Message, PromptTemplate
from torchtune.models.llama2._prompt_template import Llama2ChatTemplate
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform
from transformers import LlamaTokenizer

PROMPT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}\n{% for message in loop_messages %}\n{% if message['role'] not in ['user', 'assistant', 'tool_calls'] %}\n{{ raise_exception('Invalid role: ' + message['role'] + '. Must be user, assistant, or tool_calls.') }}\n{% endif %}\n{% if loop.index0 == 0 and system_message != false %}\n{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}\n{% else %}\n{% set content = message['content'] %}\n{% endif %}\n{% if message['role'] == 'user' %}\n{{ '<s>[INST] ' + content.strip() + ' [/INST]' }}\n{% elif message['role'] == 'assistant' %}\n{{ ' ' + content.strip() + ' </s>' }}\n{% elif message['role'] == 'tool_calls' %}\n{{ ' [TOOL_CALLS] ' + content.strip() + ' [/TOOL_CALLS] ' }}\n{% endif %}\n{% endfor %}"


class Sarvam1Tokenizer(ModelTokenizer, Transform):
    """
    This is the same as the Llama2Tokenizer, but with special handling for the spiece_underline token.

    Args:
        path (str): Path to pretrained SentencePiece tokenizer file.
        max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
            Default: None
        prompt_template (Optional[PromptTemplate]): template used to format the messages based on their role. This is used
            to add structured text around the actual messages. The structured text is used in three scenarios:

            - Task-specific templates to gear models for a particular task that it will expect after training
            - Model-specific templates that are required whenever the model is prompted, such as the [INST]
              tags in Llama2 and in Mistral
            - Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

            The extra text will still get tokenized as normal text, not as special tokens.
            Default is :class:`~torchtune.models.llama2.Llama2ChatTemplate`.

    Examples:
        >>> tokenizer = Sarvam1Tokenizer("/path/to/spm_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[str] = Llama2ChatTemplate(),
    ):
        if not path.endswith('.model'):
            raise ValueError(f"Tokenizer path must end with '.model', got {path}")
        self._tokenizer = LlamaTokenizer(vocab_file=path, legacy=False, split_special_tokens=True)
        self._tokenizer.pad_id = self.eos_id if self.pad_id is None else self.pad_id
        self.stop_tokens = [self.eos_id]
        self._tokenizer.chat_template = PROMPT_TEMPLATE
        self.max_seq_len = max_seq_len
        self.shown_tokenize_messages_warning = False
    
    @property
    def eos_id(self):
        return self._tokenizer.eos_token_id

    @property
    def bos_id(self):
        return self._tokenizer.bos_token_id

    @property
    def pad_id(self):
        return self._tokenizer.pad_token_id

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode a string into a list of tokens.
        Note:
            Currently this method does not add eos token, and does not trim leading whitespace.
        """
        tokens = self._tokenizer.encode(text, add_special_tokens=add_bos)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens
    
    def decode(
        self,
        token_ids: List[int],
    ) -> str:
        return self._tokenizer.decode(token_ids)
    
    def create_assistant_mask(self, templated_message):
        tokens = self._tokenizer.tokenize(templated_message)
        mask = [1] * len(tokens)

        is_assistant = False
        for i in range(len(tokens)):
            if tokens[i-1] == '[/INST]' and tokens[i] == '\n':
                is_assistant = True
                continue
            if tokens[i-1] == '</s>' and tokens[i] == '\n':
                is_assistant = False
                continue
            if is_assistant:
                mask[i] = 0
        return mask

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_bos_tokens: bool = False,
        add_end_tokens: bool = False,
    ) -> Tuple[List[int], List[bool]]:
        r"""Tokenize a list of messages one at a time then concatenate them,
        returning a list of tokens and a list of masks.

        Note:
            sentencepiece has problems where in general
            encode(s1 + s2) != encode(s1) + encode(s2) due to whitespace handling.
            We can get around this by prepending s2 with a known token and slicing the
            beginning off the tokenized s2.

        Example:
            >>> tokenizer = Sarvam1Tokenizer(tokenizer_path, max_seq_len)
            >>> messages = [
                Message(role="system", content="system message\n", masked=True),
                Message(role="user", content="user prompt\n", masked=True),
                Message(role="assistant", content="assistant response\n"),
            ]

            >>> # tokenize_messages encodes messages separately and concats
            >>> tokenizer.tokenize_messages(messages)[0]
            [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]

            >>> # Same result as encoding the full string in one go
            >>> tokenizer.encode(''.join([message.content for message in messages]))
            [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]


        Args:
            messages (List[Message]): A list of messages, each containing role, content,
                and masked attributes.

        Returns:
            Tuple[List[int], List[bool]]: The tokenized messages
        """
        if add_bos_tokens or add_end_tokens and not self.shown_tokenize_messages_warning:
            print("WARNING: You have passed `add_bos_tokens` or `add_end_tokens` to `tokenize_messages`.")
            print("WARNING: This will change the behavior of the tokenizer. Both arguments will be ignored.")
            self.shown_tokenize_messages_warning = True
        hf_messages = []
        for message in messages:
            hf_messages.append({"role": message.role, "content": message.content[0]["content"]})
        templated_message = self._tokenizer.apply_chat_template(hf_messages, tokenize=False)
        input_ids = self._tokenizer(templated_message, add_special_tokens=False)["input_ids"]
        attention_mask = [bool(x) for x in self.create_assistant_mask(templated_message)]
        return input_ids, attention_mask
    
    def __call__(
        self, sample: Mapping[str, Any], inference: bool = False
    ) -> Mapping[str, Any]:
        """
        Apply ``tokenize_messages`` to the "messages" field in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with a "messages" field containing
                a List[Message] to tokenize
            inference (bool): Whether the template is being used for inference or not.

        Returns:
            Mapping[str, Any]: The sample with added "tokens" and "mask" fields
                and the "messages" field removed.
        """
        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages, add_end_tokens=not inference)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample