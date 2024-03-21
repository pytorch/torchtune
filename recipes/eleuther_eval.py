# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time

from functools import partial
from typing import Any, Dict, Optional, Tuple
from warnings import warn

import torch
from omegaconf import DictConfig

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from torchtune import config, modules, utils
from torchtune.modules import Tokenizer, TransformerDecoder

from torchtune.recipe_interfaces import EvalRecipeInterface

from tqdm import tqdm


logger = utils.get_logger("DEBUG")

try:
    import lm_eval
    from lm_eval.evaluator import evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import get_task_dict
except (ImportError, ModuleNotFoundError) as e:
    logger.error(
        f"Recipe requires EleutherAI Eval Harness v0.4. Please install with `pip install lm_eval==0.4.*`"
    )
    print(e)
    sys.exit(1)

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
        super().__init__(device=str(device))
        self._model = model
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id

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

    def tok_decode(self, tokens, **kwargs):
        decoded = self._tokenizer.decode(tokens)
        return decoded

    def _model_call(self, inps):
        logits = self._model(inps)
        return logits

    def _model_generate(self, *args, **kwargs):
        raise Exception('unimplemented')


class EleutherEvalRecipe(EvalRecipeInterface):
    """

    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)
        self.seed = utils.set_seed(seed=cfg.seed)
        self._limit = cfg.limit
        self._tasks = list(cfg.tasks)


    def load_checkpoint(self, cfg: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        self._checkpointer = config.instantiate(
            cfg,
            resume_from_checkpoint=False,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        return checkpoint_dict

    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up the recipe state correctly. This includes setting recipe attributes based
        on the ``resume_from_checkpoint`` flag.
        """

        ckpt_dict = self.load_checkpoint(cfg.checkpointer)

        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._model = self._setup_model(
            cfg_model=cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)
        logger.info("Tokenizer is initialized from file.")

    def _setup_model(
        self,
        cfg_model: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        """
        Set up the model.
        """
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model, dtype=self._dtype)
        logger.info(f"Model is initialized with precision {self._dtype}.")
        return model

    @torch.no_grad()
    def evaluate(self) -> None:
        t1 = time.time()

        model_eval_wrapper = _EvalWrapper(
            self._model,
            self._tokenizer,
            self._device,
            max_seq_length=4096,
        )

        try:
            lm_eval.tasks.initialize_tasks()
        except:
            pass

        task_dict = get_task_dict(self._tasks or _default_tasks)
        eleuther_output = evaluate(
            model_eval_wrapper,
            task_dict,
            limit=self._limit,
        )

        # Report results.
        logger.info(f"Time to run eval: {time.time() - t1:.02f} seconds.")
        for task, res in eleuther_output["results"].items():
            print(f"{task}: {res}")


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.
    """
    recipe = EleutherEvalRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.evaluate()


if __name__ == "__main__":
    sys.exit(recipe_main())
