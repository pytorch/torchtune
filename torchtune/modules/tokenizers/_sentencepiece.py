# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

from sentencepiece import SentencePieceProcessor
from torchtune.data._types import Message
from torchtune.data._utils import truncate


# start and end header tokens for formatting chat messages
START_HEADER_ID = "<start_of_turn>"
END_HEADER_ID = "\n"
EOT_ID = "<end_of_turn>"

PAD_ID = -1


WHITESPACE_CHARS = [" ", "\n", "\t", "\r", "\v"]



class SentencePieceTokenizer:
    """A wrapper around SentencePieceProcessor.

    Args:
        path (str): Path to pretrained tokenizer file.

    Example:
        # Accepts only non-batched input for now
        >>> tokenizer = SentencePieceTokenizer("/path/to/spm_model")
        >>> tokenized_text = SentencePieceTokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
        start_header_id: str = START_HEADER_ID,
        end_header_id :str = END_HEADER_ID,
        eot_id: str = EOT_ID,
    ):
        spm_model = SentencePieceProcessor()
        spm_model.load(path)
        self.spm_model = spm_model
        self.vocab_size = spm_model.vocab_size()
        self.bos_id = spm_model.bos_id()
        self.eos_id = spm_model.eos_id()
        self.pad_id = spm_model.pad_id()

        self.start_header_id = self.encode(start_header_id, add_bos=False, add_eos=False)[0]
        self.end_header_id = self.encode(end_header_id, add_bos=False, add_eos=False)[0]
        self.eot_id = self.encode(eot_id, add_bos=False, add_eos=False)[0]

        # This is used in tokenize_messages: if the tokenizer does not
        # encode whitespace, then we can more easily split strings
        # on whitespace characters and encode them separately.
        self.encodes_whitespace = any(
            [self.spm_model.encode(c) for c in WHITESPACE_CHARS]
        )

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        trim_leading_whitespace: bool = False,
        prefix: Optional[str] = None,
    ) -> List[int]:
        """Encode text into token IDs.

        Args:
            text (str): The input text to be encoded, unbatched.
            add_bos (bool): Whether to prepend BOS to the input, defaults to True.
            add_eos (bool): Whether to append EOS to the input, defaults to True.
            trim_leading_whitespace (bool): Whether to trim leading whitespace from
                underlying sentencepiece tokenization. Sentencepiece normally prepends
                whitespace to any tokenized text, which can cause differences where
                encode(s1) + encode(s2) != encode(s1 + s2) due to leading whitespace
                added to s2. Default: False
            prefix (Optional[str]): Optional string to encode for trimming leading
                whitespaces. Used only if trim_leading_whitespace=True. Default: None
        Returns:
            List[int]: The encoded token IDs.
        """
        if trim_leading_whitespace:
            # Can define our own custom prefix depending on vocab if needed
            if not hasattr(self, "prefix"):
                self.prefix = prefix or "\n"
                self.encoded_prefix = self.spm_model.encode(
                    self.prefix, add_bos=False, add_eos=False
                )
            start_idx = len(self.encoded_prefix) + int(add_bos)
            return self.spm_model.encode(
                self.prefix + text,
                add_bos=add_bos,
                add_eos=add_eos,
                out_type=int,
            )[start_idx:]
        else:
            return self.spm_model.encode(
                text,
                add_bos=add_bos,
                add_eos=add_eos,
                out_type=int,
            )
    def encode_with_special_tokens(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        trim_leading_whitespace: bool = False,
        prefix: Optional[str] = None,
    ) -> List[int]:

        return self.encode(
            text=text,
            add_bos=add_bos,
            add_eos=add_eos,
            trim_leading_whitespace=trim_leading_whitespace,
            prefix=prefix,
        )

    def decode(self, ids: List[int], truncate_at_eos: bool = True) -> str:
        """Decode token IDs to strings.

        Args:
            ids (List[int]): The input token IDs to be decoded.

        Returns:
            str: The decoded text.
        """
        if truncate_at_eos:
            try:
                k = ids.index(self.eos_id)
            except ValueError:
                k = None
            if k:
                ids = ids[:k]

        return self.spm_model.decode(ids)

    def tokenize_messages(
        self, messages: List[Message],
        max_seq_len: Optional[int] = None,
        tokenize_header: bool = True,
        unmask_outputs: bool = False,
        chat_format: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        r"""Tokenize a list of messages one at a time then concatenate them,
        returning a list of tokens and a list of masks.

        Note: llama2 sentencepiece has problems where in general
        encode(s1 + s2) != encode(s1) + encode(s2) due to whitespace handling.
        We can get around this by prepending s2 with a known token and slicing the
        beginning off the tokenized s2.

        Example:
            >>> tokenizer = SentencePieceTokenizer(tokenizer_path)
            >>> messages = [
                Message(role="system", content="system message\n", masked=True),
                Message(role="user", content="user prompt\n", masked=True),
                Message(role="assistant", content="assistant response\n"),
                ]
            # tokenize_messages encodes messages separately and concats
            >>> tokenizer.tokenize_messages(messages, max_seq_len)[0]
            [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]


            # Same result as encoding the full string in one go
            >>> tokenizer.encode(''.join([message.content for message in messages]))
            [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]


        Args:
            messages (List[Message]): A list of messages, each containing role, content,
                and masked attributes.
            max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
                Default: None

        Returns:
            Tuple[List[int], List[bool]]: The tokenized messages
        """
        tokenized_messages = [self.bos_id]
        # bos and eos are always masked
        mask = [True]
        if not chat_format:
            messages[0].content = "".join([message.content for message in messages])
            messages = messages[:1]



        for i, message in enumerate(messages):
            # If assistant message, this is the end of a turn
            if chat_format and tokenize_header:
                tokenized_header = (
                    [self.start_header_id]
                    + self.encode(message.role.strip(), add_bos=False, add_eos=False)
                    + [self.end_header_id]
                    # + self.encode("\n\n", add_bos=False, add_eos=False)
                )
            else:
                tokenized_header = []

            tokenized_body = self.encode(
                message.content.strip(), add_bos=False, add_eos=False
            )
            if chat_format:
                # if message.ipython:
                #     tokenized_body = [self.python_tag] + tokenized_body
                tokenized_message = tokenized_header + tokenized_body
                #if message.eot:
                tokenized_message = tokenized_message + [self.eot_id]

                if i < len(messages) - 1:
                    tokenized_message = tokenized_message + [self.end_header_id]
                #else:
                #    tokenized_message = tokenized_message + [self.eot_id]
            else:
                tokenized_message = tokenized_body

            # Prepend BOS on start of new turns
            # if start_of_turn:
            #     tokenized_messages.append(self.bos_id)
            #     mask.append(message.masked)

            # We want to trim leading whitespace on the next message when
            # (a) it is a continuation of the turn (i.e. not the first message)
            # (b) the vocabulary explicitly encodes whitespace characters, and
            # (c) the previous message did not end with a space
            # trim_leading_whitespace = (
            #     (not start_of_turn)
            #     and self.encodes_whitespace
            #     and not prev_ends_with_space
            # )

            # # Tokenize current message, append with masks
            # tokens = self.encode(
            #     message.content.rstrip(" "),
            #     add_bos=False,
            #     add_eos=False,
            #     trim_leading_whitespace=trim_leading_whitespace,
            # )
            # prev_ends_with_space = message.content.endswith(" ")
            tokenized_messages.extend(tokenized_message)

            if unmask_outputs and message.role == "system" and "code" not in message.content:
                # we want to mask outputs after first example in the sequence
                # find second all positions of -> token 1492
                # fast find all 1492 in tokenized_message
                all_sep_positions = [i for i, x in enumerate(tokenized_message) if x in (3978, 949)]
                # find all ]]
                all_close_positions = [i for i, x in enumerate(tokenized_message) if x in (10761, 97658)]
                mask_for_system = [True] * len(tokenized_message)
                if len(all_sep_positions) > 1:
                    for sep_position in all_sep_positions[1:]:
                        # find the next close bracket
                        close_position = [x for x in all_close_positions if x > sep_position][0]
                        mask_for_system[sep_position+1:close_position+1] = [False] * (close_position - sep_position)
                # mask positions 2 - 3
                mask = mask + mask_for_system
            else:
                mask = mask + ([message.masked] * len(tokenized_message))

            if max_seq_len and len(tokenized_messages) >= max_seq_len:
                break

        # If assistant message, append EOS at end
        tokenized_messages.append(self.eos_id)
        mask.append(True)


        # Finally, truncate if necessary
        if max_seq_len:
            tokenized_messages = truncate(tokenized_messages, max_seq_len, self.eos_id)
            mask = truncate(mask, max_seq_len, True)

        return tokenized_messages, mask
