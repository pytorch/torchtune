# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, fields


@dataclass
class AlpacaGenerateParams:
    """Arguments for the alpaca_generate recipe.

    Args:
        model (str): String specifying model architecture to run inference with. See ``torchtune.models.get_model`` for options.
        model_checkpoint (str): Local path to load model checkpoint from.
        tokenizer (str): String specifying tokenizer to use. See ``torchtune.models.get_tokenizer`` for options.
        tokenizer_checkpoint (str): Local path to load tokenizer checkpoint from.
        instruction (str): Instruction for model to respond to.
        input (str): Additional optional input related to instruction. Pass in "" (empty string) for no input.
        max_gen_len (int): Max number of tokens to generate
    """

    # Model
    model: str = ""
    model_checkpoint: str = ""

    # Tokenizer
    tokenizer: str = ""
    tokenizer_checkpoint: str = ""

    # Generation
    instruction: str = ""
    input: str = ""
    max_gen_len: int = 64

    def __post_init__(self):
        for param in fields(self):
            if getattr(self, param.name) == "" and param.name != "input":
                raise TypeError(f"{param.name} needs to be specified")
