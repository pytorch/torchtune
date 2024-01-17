# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# python -m recipes.evaluate --native-checkpoint-path /tmp/finetune-llm/model_2.ckpt --tokenizer-path ~/llama/tokenizer.model

import argparse
import logging
import time
from typing import List, Optional, Tuple

import lm_eval
import torch
import torch._dynamo.config
import torch._inductor.config
from lm_eval.api.instance import Instance

from torchtune.models.llama2 import llama2_7b, llama2_tokenizer
from torchtune.modules.tokenizer import Tokenizer
from torchtune.modules.transformer import TransformerDecoder
from torchtune.utils.device import _get_device_from_env
from torchtune.utils.env import seed
from torchtune.utils.generation import GenerationUtils

from .base_lm import BaseLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvalWrapper(BaseLM):
    """
    A wrapper class for TransformerDecoder, providing integration with the lm-evaluation-harness library.
    """

    def __init__(
        self,
        model: TransformerDecoder,
        tokenizer: Tokenizer,
        max_seq_length: Optional[int] = None,
    ):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device("cuda")
        self._max_seq_length = 2048 if max_seq_length is None else max_seq_length

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id()

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 50

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        encoded = self._tokenizer.encode(string, add_eos=False)
        return encoded

    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def tok_decode(self, tokens):
        decoded = self._tokenizer.decode(tokens)
        return decoded

    def _model_call(self, inps):
        logits = self._model(inps)

        return logits

    def _model_generate(self, context, max_length, eos_token_id):
        generations_no_kv_cache, _ = GenerationUtils(
            decoder_lm=self._model,
            eos_id=eos_token_id,
            pad_id=self._tokenizer.pad_id,
        ).generate(
            prompt_tokens=context,
            incremental_decode=False,
            min_gen_len=1,
            max_gen_len=max_length,
            top_k=3,
            device=torch.cuda.current_device(),
        )
        gens = generations_no_kv_cache.tolist()[0]
        return gens

    def generate_until(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("`generate_until` is not implemented.")


def main(
    native_checkpoint_path: str,
    tasks: List[str] = ["hellaswag"],  # noqa: B006
    limit: Optional[int] = None,
    max_seq_length: Optional[int] = None,
) -> None:
    """Evaluates model on a task from the `lm-evaluation-harness` library.

    Args:
        native_checkpoint_path (str): The path to the model checkpoint file to load.
        tasks (List[str]): The name of the evaluation task or a list of tasks to perform.
        limit (Optional[int]): The maximum number of samples to evaluate (None for all available).
        max_seq_length (Optional[int]): The maximum sequence length allowed for input text.

    """

    tokenizer = llama2_tokenizer(args.tokenizer_path)

    seed(1234)

    device = _get_device_from_env()
    # --------- Initialize a decoder w/o kv-caching -------- #
    with device:
        model = llama2_7b()

    # Load state_dict into model
    native_state_dict = torch.load(args.native_checkpoint_path, weights_only=True)
    # Note: If using pretrained model, replace native_state_dict["model"] with native_state_dict
    missing, unexpected = model.load_state_dict(
        native_state_dict["model"], strict=False
    )

    model.eval()

    print("Loading model ...")
    t0 = time.time()

    torch.cuda.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds.")

    t1 = time.time()
    with torch.no_grad():
        model_eval_wrapper = ModelEvalWrapper(
            model,
            tokenizer,
            max_seq_length,
        )

        lm_eval.tasks.initialize_tasks()
        task_dict = lm_eval.tasks.get_task_dict(tasks)

        eval_results = lm_eval.evaluator.evaluate(
            model_eval_wrapper,
            task_dict,
            limit=limit,
        )
    print(f"Time to run eval: {time.time() - t1:.02f} seconds.")
    print(f"For model {args.native_checkpoint_path}")
    for task, res in eval_results["results"].items():
        print(f"{task}: {res}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a finetuned model.")

    parser.add_argument(
        "--native-checkpoint-path", type=str, help="Path to native checkpoint file."
    )
    parser.add_argument("--tokenizer-path", type=str, help="Path to tokenization file.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["mmlu"],
        help="list of lm-eleuther tasks to evaluate usage: --tasks task1 task2",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="number of samples to evalulate"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="maximum length sequence to evaluate",
    )

    args = parser.parse_args()
    main(
        args.native_checkpoint_path,
        args.tasks,
        args.limit,
        args.max_seq_length,
    )
