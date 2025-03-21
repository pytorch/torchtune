# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any, Dict, List, Optional, Union
import json

import torch
from omegaconf import DictConfig
from torch import nn

from torchtune import config, generation, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import ChatFormat, InstructTemplate, Message
from torchtune.training import FullModelTorchTuneCheckpointer

import pandas as pd

logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)

        training.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)

        ckpt_dict = checkpointer.load_checkpoint()

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[training.MODEL_KEY],
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )
        logger.info(f"Model is initialized with precision {self._dtype}.")

        return model

    @torch.inference_mode()
    def generate(self, cfg: DictConfig):
        with open(cfg.validation_set) as f:
            validation_set = [json.loads(line) for line in f]

        system_prompt = cfg['dataset']['new_system_prompt']
        total_gen_accuracy = 0

        results = pd.DataFrame(columns=[
            'example_idx',
            'input',
            'model_output',
            'ground_truth',
            'classification']
        )

        for i, example in enumerate(validation_set):
            if i >= cfg.num_examples: break
            ground_truth = example['output']

            msgs = [Message(role="system", content=system_prompt, eot=True), Message(role="user", content=example['input'], eot=True)]
            tokens, mask = self._tokenizer.tokenize_messages(msgs)
            tokens = tokens[:-1] # remove EOS because we want to continue generating
            prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)

            generated_tokens, _ = generation.generate(
                model=self._model,
                prompt=prompt,
                max_generated_tokens=cfg.max_new_tokens,
                pad_id=self._tokenizer.pad_id,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                stop_tokens=self._tokenizer.stop_tokens,
                custom_generate_next_token=None,
            )
            start_idx = torch.where(prompt == self._tokenizer.eot_id)[0][1]

            output = self._tokenizer.decode(generated_tokens[0][start_idx:].tolist())

            pass_text = 'compliance output: pass'
            fail_text = 'compliance output: fail'
            classification = 'null'

            if pass_text in output.lower() and pass_text in ground_truth.lower():
                classification = 'true_pass'
            if pass_text in output.lower() and fail_text in ground_truth.lower():
                classification = 'false_pass'
            if fail_text in output.lower() and fail_text in ground_truth.lower():
                classification = 'true_fail'
            if fail_text in output.lower() and pass_text in ground_truth.lower():
                classification = 'false_fail'

            if classification == 'true_pass' or classification == 'true_fail':
                total_gen_accuracy += 1

            results.loc[len(results.index)] = [i, example['input'], output, ground_truth, classification]
            print(i, classification)
        print(f'Total accuracy: {total_gen_accuracy} / {max(len(validation_set), cfg.num_examples)}')
        results.to_csv('validation_results.csv')

@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
