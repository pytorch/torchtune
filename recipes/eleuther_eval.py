# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time

from typing import Any, Dict, List, Tuple, Union

import torch
from omegaconf import DictConfig

from torch import nn
from torch.nn.utils.rnn import pad_sequence

from torchtune import config, utils
from torchtune.modules import TransformerDecoder
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.recipe_interfaces import EvalRecipeInterface


logger = utils.get_logger("DEBUG")

try:
    import lm_eval
    from lm_eval.evaluator import evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import get_task_dict, TaskManager
    from lm_eval.utils import make_table
except ImportError:
    logger.error(
        "Recipe requires EleutherAI Eval Harness v0.4. Please install with `pip install lm_eval==0.4.*`"
    )
    sys.exit(1)


class _EvalWrapper(HFLM):
    """An EvalWrapper for EleutherAI's eval harness based on gpt-fast's
    EvalWrapper: https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py.

    Args:
        model (TransformerDecoder): The model to evaluate.
        tokenizer (ModelTokenizer): Tokenizer associated with the model being evaluated.
            This should be the same tokenizer used when fine-tuning the model.
        device (torch.device): The device to use.
        max_seq_length (int): The maximum sequence length to use.
        batch_size (int): The batch size per GPU to use.
    """

    def __init__(
        self,
        model: TransformerDecoder,
        tokenizer: ModelTokenizer,
        *,
        device: torch.device,
        max_seq_length: int = 4096,
        batch_size: int = 8,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(pretrained="gpt2", device=str(device))
        self._model = model
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._dtype = dtype

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, text: str, **kwargs) -> List[int]:
        # Note on add_bos flag: setting to False as this gives better results, for example
        # +1% on truthfulqa_mc2 with a LoRA finetune. lit-gpt also sets this to False,
        # see https://github.com/Lightning-AI/lit-gpt/blob/main/eval/lm_eval_harness.py#L66,
        # though notably fast-gpt does the opposite
        # https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py#L123.
        return self._tokenizer.encode(text=text, add_bos=False, add_eos=False)

    def tok_batch_encode(
        self, text: List[str], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized_text = [self.tok_encode(x) for x in text]

        # pad left
        x = pad_sequence(
            [
                torch.tensor(x[::-1]) for x in tokenized_text
            ],  # first flip each sequence and pad
            batch_first=True,
            padding_value=self._tokenizer.pad_id,
        ).flip(
            dims=[1]
        )  # flip back to correct order

        return x, torch.ones_like(x)  # return 'mask' b/c it's expected by the harness

    def tok_decode(self, tokens: Union[List[int], int], **kwargs) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._model(inps)

    def _model_generate(
        self, context: torch.Tensor, **generation_kwargs
    ) -> torch.Tensor:
        curr_batch_size = context.size(0)

        # Setup caches for a given batch size
        # Technically this is not necessary, but it's a good way to ensure that
        # the caches won't error on a different batch size. In addition, caches
        # are not needed for a regular model call, so we just setup here
        with context.device:
            self._model.setup_caches(batch_size=curr_batch_size, dtype=self._dtype)

        temperature = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", False)
        if do_sample:
            # do_sample signifies more complicated sampling logic, like top_k or
            # top_p. We don't support this yet, so if it's requested, we raise an error.
            raise RuntimeError(
                "``do_sample`` for generation tasks is not supported yet in torchtune."
            )

        toks = utils.generate(
            self._model,
            context,
            max_generated_tokens=self.max_gen_toks,
            pad_id=self._tokenizer.pad_id,
            temperature=temperature,
            top_k=None,  # do_sample is not supported currently
            stop_tokens=self._tokenizer.stop_tokens,
        )
        return torch.tensor(toks, dtype=torch.int32)


class EleutherEvalRecipe(EvalRecipeInterface):
    """
    This recipe runs evaluation on a trained model using EleutherAI's eval harness.
    This assumes the user has the EleutherAI eval harness installed. See
    https://github.com/EleutherAI/lm-evaluation-harness for more details.

    Features:
        - Single GPU evaluation. Multi-GPU evaluation is currently not supported.
        - Loading model in fp32 or bf16. Fp16 is currently not supported.
        - Any task from the EleutherAI eval harness that is *not* free generation

    We recommend launching evaluation using the tune CLI:

        tune run eleuther_eval --config llama2_eleuther_eval \
        tasks=["truthfulqa_mc2","hellaswag"]

    Args:
        cfg (DictConfig): OmegaConf object parsed from YAML file
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def setup(self) -> None:
        self._device = utils.get_device(device=self._cfg.device)
        self._dtype = utils.get_dtype(dtype=self._cfg.dtype)
        self._limit = self._cfg.limit
        self._tasks = list(self._cfg.tasks)
        self._quantizer = config.instantiate(self._cfg.quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)

        utils.set_seed(seed=self._cfg.seed)

        checkpointer = config.instantiate(self._cfg.checkpointer)
        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        self._model = self._setup_model(
            model_cfg=self._cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
        )
        self._tokenizer = config.instantiate(self._cfg.tokenizer)
        logger.info("Tokenizer is initialized from file.")

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)
        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)

        model.load_state_dict(model_state_dict)
        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        logger.info(f"Model is initialized with precision {self._dtype}.")
        return model

    @torch.no_grad()
    def evaluate(self) -> None:
        t1 = time.time()

        model_eval_wrapper = _EvalWrapper(
            self._model,
            self._tokenizer,
            device=self._device,
            max_seq_length=self._cfg.max_seq_length,
            batch_size=self._cfg.batch_size,
            dtype=self._dtype,
        )

        # Task initialization API changed between v0.4.1 and 0.4.2
        try:
            lm_eval.tasks.initialize_tasks()
        except Exception:
            pass

        task_manager = TaskManager(include_path=self._cfg.get("include_path", None))
        task_dict = get_task_dict(self._tasks, task_manager)

        logger.info(f"Running evaluation on {self._tasks} tasks.")
        output = evaluate(
            model_eval_wrapper,
            task_dict,
            limit=self._limit,
        )

        logger.info(f"Eval completed in {time.time() - t1:.02f} seconds.")

        formatted_output = make_table(output)
        print(formatted_output)


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entry point for the recipe."""
    config.log_config(recipe_name="EleutherEvalRecipe", cfg=cfg)
    recipe = EleutherEvalRecipe(cfg=cfg)
    recipe.setup()
    recipe.evaluate()


if __name__ == "__main__":
    sys.exit(recipe_main())
