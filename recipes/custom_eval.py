# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchtune.utils as utils
from omegaconf import DictConfig
from torchtune import config
from torchtune.recipe_interfaces import EvalRecipeInterface


class CustomEvalRecipe(EvalRecipeInterface):
    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)
        self.seed = utils.set_seed(seed=cfg.seed)

    def load_checkpoint(self, cfg: DictConfig) -> Dict[str, Any]:
        self._checkpointer = config.instantiate(
            cfg,
            resume_from_checkpoint=False,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        return checkpoint_dict

    def _setup_model(
        self,
        cfg_model: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with self._device:
            model = config.instantiate(cfg_model)

        model.load_state_dict(model_state_dict)

        log.info("Model is initialized.")
        return model

    def setup(self, cfg: DictConfig) -> None:
        ckpt_dict = self.load_checkpoint(cfg.checkpointer)

        self._model = self._setup_model(
            cfg_model=cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized.")

        self._dataloader = config.instantiate(cfg.dataset)

    @torch.no_grad()
    def evaluate(self) -> None:
        # Accumulate accuracy over all batches
        total_acc = 0.0

        for idx, batch in enumerate(pbar := tqdm(self._dataloader)):
            pbar.set_description(f"Batch {idx}")

            input_ids, labels = batch
            input_ids = input_ids.to(self._device)
            labels = labels.to(self._device)

            # Forward pass
            outputs = self._model(input_ids=input_ids)

            # Compute accuracy between predicted generation and labels
            acc = torch.sum(outputs.argmax(dim=-1) == labels) / labels.size(0)
            total_acc += acc

        # Average accuracy over all batches
        avg_acc = total_acc / len(self._dataloader)
        log.info(f"Accuracy: {avg_acc}")


def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in ``alpaca_llama2_full_finetune.yaml``
        - Overwritten by arguments from the command-line
    """
    recipe = CustomEvalRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.evaluate()


if __name__ == "__main__":
    sys.exit(recipe_main())
