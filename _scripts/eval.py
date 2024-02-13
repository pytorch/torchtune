# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import time
from typing import Any, Dict, List, Optional

import lm_eval

import torch
import torchtune.utils as utils
from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict
from torchtune import models
from torchtune.modules import Tokenizer, TransformerDecoder
from torchtune.utils import get_logger

logger = get_logger("DEBUG")

_default_tasks = ["hellaswag"]


class _EvalWrapper(HFLM):
    """
    An EvalWrapper for EleutherAI's eval harness based on fast-gpt's
    EvalWrapper: https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device_str,
        max_seq_length,
    ):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = utils.get_device(device=device_str)
        self._max_seq_length = max_seq_length

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id()

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 512

    @property
    def batch_size(self):
        return 32

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):
        encoded = self._tokenizer.encode(text=string, add_bos=False, add_eos=False)
        return encoded

    def tok_decode(self, tokens):
        decoded = self._tokenizer.decode(tokens)
        return decoded

    def _model_call(self, inps):
        logits = self._model(inps)
        return logits


@torch.no_grad()
def eval(
    model: TransformerDecoder,
    tokenizer: Tokenizer,
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
    max_seq_length: Optional[int] = 4096,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    """
    Evaluates a language model on a specified task using the lm-evaluation-harness library. Based
    off of fast-gpt's evaluation wrapper: https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py

    Args:
        model (TransformerDecoder): The pre-trained or finetuned language model to evaluate.
        tokenizer (Tokenizer): The tokenizer to use for encoding/decoding text.
        tasks (Optional[List[str]]): The names of the evaluation tasks to perform. Defaults to ``None``,
            in which case "hellaswag" task is evaluated on.
        limit (Optional[int]): The maximum number of samples to evaluate (None for all available).
        max_seq_length (Optional[int]): The maximum sequence length allowed for input text. Defaults to 4096.
        device_str (str): String indicating device to run eval on. Can be "cpu", "cuda", or "cuda:<device_id>".

    Returns:
        eval_results (Dict[str, Any]): A dictionary of evaluation results for the specified task(s).
    """

    if tasks is None:
        tasks = _default_tasks

    model_eval_wrapper = _EvalWrapper(
        model,
        tokenizer,
        device_str,
        max_seq_length,
    )

    lm_eval.tasks.initialize_tasks()

    task_dict = get_task_dict(tasks)

    eval_results = evaluate(
        model_eval_wrapper,
        task_dict,
        limit=limit,
    )
    return eval_results


def main(
    model_name: str,
    model_checkpoint: str,
    tokenizer_name: str,
    tokenizer_checkpoint: str,
    device: str = "cuda",
    task_list: List[str] = None,
    limit: Optional[int] = None,
) -> None:
    """
    Instantiates a model and tokenizer from checkpoints and evaluates the model on the specified
    ``task_list`` using the lm-evaluation-harness library (https://github.com/EleutherAI/lm-evaluation-harness).

    Args:
        model_name (str): The name of the model to evaluate.
        model_checkpoint (str): The path to the model checkpoint.
        tokenizer_name (str): The name of the tokenizer to use.
        tokenizer_checkpoint (str): The path to the tokenizer checkpoint.
        device (str): The device to run evaluation on. Can be "cpu", "cuda", or "cuda:<device_id>"
        task_list (List[str]): The names of the evaluation tasks to perform. Defaults to ``None``,
            in which case "hellaswag" task will be used.
        limit (Optional[int]): The maximum number of samples to evaluate (None for all available).
    """

    torch.manual_seed(1234)

    if task_list is None:
        task_list = _default_tasks
    # Create tokenizer and model.
    tokenizer = models.get_tokenizer(tokenizer_name, path=tokenizer_checkpoint)
    model = models.get_model(model_name, device=device)
    model_state_dict = torch.load(
        f=model_checkpoint,
        weights_only=True,
        map_location="cpu",
    )
    model.load_state_dict(model_state_dict["model"])
    logger.info(
        f"Initialized model and tokenizer from {model_checkpoint} and {tokenizer_checkpoint}. Running eval."
    )
    t1 = time.time()
    result = eval(
        model=model,
        tokenizer=tokenizer,
        tasks=task_list,
        limit=limit,
        device_str=device,
    )
    logger.info(f"Time to run eval: {time.time() - t1:.02f} seconds.")
    logger.info(f"For model {model_checkpoint}")
    for task, res in result["results"].items():
        logger.info(f"{task}: {res}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EleutherAI LLM Evaluation Harness.")
    parser.add_argument(
        "--model",
        type=str,
        default="llama2_7b",
        choices=models.list_models(),
        help="Name of the model to finetune.",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="/tmp/llama2-7b",
        help="Path to native checkpoint file.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="llama2_tokenizer",
        choices=models.list_tokenizers(),
        help="Name of the model tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        default="/tmp/tokenizer.model",
        help="Path to tokenization file.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["hellaswag"],
        help="list of lm-eluther tasks to evaluate usage: --tasks task1 task2",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="number of samples to evalulate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for eval. Can be cpu, cuda, or cuda:<device_id>",
    )
    args = parser.parse_args()
    main(
        model_name=args.model,
        model_checkpoint=args.model_checkpoint,
        tokenizer_name=args.tokenizer,
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        device=args.device,
        task_list=args.tasks,
        limit=args.limit,
    )
