# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional
from torchtune.data import ARCInstructTemplate, ChatFormat, ChatMLFormat, arc_to_messages
from torchtune.datasets._chat import ChatDataset
from torchtune.modules.tokenizers import Tokenizer


# def arc_dataset(
#     tokenizer: Tokenizer,
#     source: str = "/raid/lingo/akyurek/git/arc/re_arc/4llama/",
#     train_on_input: bool = True,
#     max_seq_len: int = 8192,
#     split="train",
# ) -> InstructDataset:
#     """
#     Support for ARC training.

#     Masking of the prompt during training is controlled by the `train_on_input` flag, which is
#     set to `True` by `default <https://github.com/tloen/alpaca-lora/blob/main/finetune.py#L49>`_
#     - If `train_on_input` is True, the prompt is used during training and
#     contributes to the loss.
#     - If `train_on_input` is False, the prompt is masked out (tokens replaced with -100)

#     Args:
#         tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
#         source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`.
#         train_on_input (bool): Whether the model is trained on the prompt or not. Default is True.
#         max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
#             Default is 512, but we recommend setting this to the highest you can fit in memory and
#             is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.

#     Returns:
#         InstructDataset: dataset configured with source data and template


#     Example:
#         >>> alpaca_ds = alpaca_dataset(tokenizer=tokenizer)
#         >>> for batch in Dataloader(alpaca_ds, batch_size=8):
#         >>>     print(f"Batch size: {len(batch)}")
#         >>> Batch size: 8
#     """

#     return InstructDataset(
#         tokenizer=tokenizer,
#         source=source,
#         template=ARCInstructTemplate,
#         train_on_input=train_on_input,
#         max_seq_len=max_seq_len,
#         split=split,
#         data_files = {"train": "augmented_train_tasks.jsonl", "test": "arc_test_tasks.jsonl"}
#     )



def arc_dataset(
    tokenizer: Tokenizer,
    source: str = "/raid/lingo/akyurek/git/arc/data/tasks/w_color_N40_rand_transformation_black_all/",
    chat_format: Optional[ChatFormat] = None,
    train_on_input: bool = True,
    unmask_outputs: bool = False,
    max_seq_len: int = 8192,
    is_chat_format_enabled: bool = True,
    split="train",
) -> ChatDataset:
    """
    Support for ARC training.

    Masking of the prompt during training is controlled by the `train_on_input` flag, which is
    set to `True` by `default <https://github.com/tloen/alpaca-lora/blob/main/finetune.py#L49>`_
    - If `train_on_input` is True, the prompt is used during training and
    contributes to the loss.
    - If `train_on_input` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is True.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
            Default is 512, but we recommend setting this to the highest you can fit in memory and
            is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.

    Returns:
        InstructDataset: dataset configured with source data and template


    Example:
        >>> alpaca_ds = alpaca_dataset(tokenizer=tokenizer)
        >>> for batch in Dataloader(alpaca_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    return ChatDataset(
        tokenizer=tokenizer,
        source=source,
        convert_to_messages=arc_to_messages,
        chat_format=chat_format,
        train_on_input=train_on_input,
        unmask_outputs=unmask_outputs,
        max_seq_len=max_seq_len,
        split=split,
        is_chat_format_enabled=is_chat_format_enabled,
        data_files = {"train": "td_False_ttd_False_ttdwa_False_ad_True_trd_False.jsonl",
                      "test": "td_True_ttd_False_ttdwa_False_ad_True_trd_False.jsonl",}
                      #"test": "arc_test_tasks.jsonl"}
                      #""test": "td_True_ttd_False_ttdwa_False_ad_True_trd_False.jsonl"}
    )



if __name__ == "__main__":
    from torchtune.modules.tokenizers import TikTokenTokenizer, SentencePieceTokenizer
    from torch.utils.data import DataLoader

    tokenizer = TikTokenTokenizer("/data/cl/scratch/model_weights/Meta-Llama-3-8B-Instruct/original/tokenizer.model")
    arc_ds = arc_dataset(tokenizer=tokenizer, is_chat_format_enabled=False, unmask_outputs=True)

    print(tokenizer.decode(arc_ds[-1][0], truncate_at_eos=False))

    tokenizer2 = SentencePieceTokenizer("/raid/lingo/models/gemma-2b-it/tokenizer.model")

    arc_ds2 = arc_dataset(tokenizer=tokenizer2, is_chat_format_enabled=False, unmask_outputs=True)

    print(tokenizer2.decode(arc_ds2[-1][0], truncate_at_eos=False))


    breakpoint()

    for batch in DataLoader(arc_ds, batch_size=8):
        print(f"Batch size: {len(batch)}")
    print("Done!")