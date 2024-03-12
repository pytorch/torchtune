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
import sys
import time
from typing import Any, Dict, List, Optional

import lm_eval

import torch
import torchtune.utils as utils
from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict
from omegaconf import DictConfig
from torchtune import config, models
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
        device,
        max_seq_length,
    ):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_seq_length = max_seq_length

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id()

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        # Sample text from model undergoing eval until this maximum output length. Using
        # 256 for now for parity with what eleuther does by default for HF models
        # (https://github.com/EleutherAI/lm-evaluation-harness/blob/96d185fa6232a5ab685ba7c43e45d1dbb3bb906d/lm_eval/models/huggingface.py#L376),
        # and we benchmark against HF llama-2 7b for correctness.
        return 256

    @property
    def batch_size(self):
        # batch size used for evaluation
        return 32

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):
        # Note on add_bos flag: setting to False as this gives better results, for example
        # +1% on truthfulqa_mc2 with a LoRA finetune. lit-gpt also sets this to False,
        # see https://github.com/Lightning-AI/lit-gpt/blob/main/eval/lm_eval_harness.py#L66,
        # though notably fast-gpt does the opposite
        # https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py#L123.
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
    device: torch.device = torch.device("cuda"),
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
        device (torch.device): torch.device indicating device to run eval on.

    Returns:
        eval_results (Dict[str, Any]): A dictionary of evaluation results for the specified task(s).
    """

    if tasks is None:
        tasks = _default_tasks

    model_eval_wrapper = _EvalWrapper(
        model,
        tokenizer,
        device,
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


def _load_checkpoint(checkpoint_path: str):
    """
    Loads a checkpoint from a given path.
    TODO: checkpoint validation.
    """
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def _setup_model(
    cfg_model: DictConfig,
    device: torch.device,
    model_checkpoint: str,
    lora_checkpoint: Optional[str] = None,
):
    """
    Sets up the model via passed in config, including loading checkpoint.
    """
    # Load state_dicts from file.
    base_model_state_dict = _load_checkpoint(model_checkpoint)
    if lora_checkpoint:
        adapter_state_dict = _load_checkpoint(lora_checkpoint)

    # Create model.
    with device:
        model = config.instantiate(cfg_model)

    # Load state_dicts into model.
    # TODO: Improve validation such as in training recipes.
    missing, unexpected = model.load_state_dict(
        base_model_state_dict["model"], strict=False
    )
    assert not unexpected, f"Unexpected keys {unexpected}"
    if missing:
        # only LoRA components can be missing. TODO: this is not robust, and can break if
        # adapter parameter names change or different PEFT techniques are used.
        for key in missing:
            assert "lora_a" in key or "lora_b" in key, f"Unexpected missing key {key}"

    if lora_checkpoint:
        lora_missing, lora_unexpected = model.load_state_dict(
            adapter_state_dict["model"], strict=False
        )
        assert not lora_unexpected, f"Unexpected keys {lora_unexpected}"
        # Intersection of missing sets should be empty.
        assert (
            lora_missing.intersection(missing) == set()
        ), f"Missing keys {lora_missing.intersection(missing)}"

    return model


@config.parse
def main(
    cfg: DictConfig,
) -> None:
    """
    WRITE DOC
    """

    # Set up environment incl. device and random seed.
    device = utils.get_device(device=cfg.device)
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    # Set up model per configuration.
    model = _setup_model(
        cfg_model=cfg.model,
        device=device,
        model_checkpoint=cfg.model_checkpoint,
        lora_checkpoint=(
            cfg.lora_checkpoint if hasattr(cfg, "lora_checkpoint") else None
        ),
    )

    # Set up tokenizer per configuration.
    tokenizer = config.instantiate(cfg.tokenizer)

    # Set up Eleuther tasks per configuration.
    if not cfg.tasks:
        task_list = _default_tasks
    else:
        task_list = cfg.tasks

    logger.info(
        f"Initialized model and tokenizer. Running eval."
    )

    # Run evaluation.
    t1 = time.time()
    result = eval(
        model=model,
        tokenizer=tokenizer,
        tasks=task_list,
        limit=cfg.limit,
        device=device,
    )
    # Report results.
    logger.info(f"Time to run eval: {time.time() - t1:.02f} seconds.")
    logger.info(f"For model {cfg.model_checkpoint}")
    for task, res in result["results"].items():
        logger.info(f"{task}: {res}")


if __name__ == "__main__":
    sys.exit(main())
