# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import DictConfig
from torch import nn

from torchtune import config, training, utils
from torchtune.data import Message, load_image, padded_collate_tiled_images_and_mask, left_pad_sequence

from torchtune.generation import sample

from torchtune.modules.transforms import Transform

logger = utils.get_logger("INFO")


class SingleTurnYAMLToMessages(Transform):
    """
    Converts a single turn conversation in YAML format to a list of messages.
    """

    def __call__(self, prompt: Dict[str, Any]) -> List[Message]:
        messages = []
        for role, content in prompt.items():
            if isinstance(content, str):
                new_content = [{"type": "text", "content": content}]
            else:
                assert "image" in content.keys(), "Multiple entries per role expect an image key"
                image_loc = content["image"]
                image = load_image(image_loc)
                new_content = [
                    {"type": "image", "content": image},
                    {"type": "text", "content": content["text"]},
                ]   
            messages.append(Message(role=role, content=new_content))
        return messages

def batch_to_device(batch: dict, device: torch.device) -> None:
    """Function that takes a dictionary (or nested dictionary) of tensors and sets them
    all to the same device. This utility is intended to be used for batches of data to be
    moved to device, the update is inplace.

    Args:
        batch (dict): dict of Tensors or more nested dicts of tensors.
        device (torch.device): torch device to move the tensor's too

    Raises:
        AttributeError: if batch dict contains anything other than tensors
    """
    for k, v in batch.items():
        if isinstance(v, dict):
            batch_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        else:
            raise AttributeError(
                "To use batch_to_device, all elements in the batch must be a dict or Tensor"
            )

class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        training.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        _checkpointer = config.instantiate(cfg.checkpointer)
        _ckpt_dict = _checkpointer.load_checkpoint()
        self.model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=_ckpt_dict[training.MODEL_KEY],
        )
        self.model_transform = config.instantiate(cfg.transform)
        self.to_messages = SingleTurnYAMLToMessages()

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        model.load_state_dict(model_state_dict)
        model.eval()

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        with self._device:
            model.setup_caches(batch_size=1, dtype=self._dtype)

        return model

    @torch.no_grad()
    def generate(self, cfg: DictConfig):
        # 1. Convert input to messages
        messages = self.to_messages(cfg.prompt)
        is_multimodal_input = any([m.contains_media for m in messages])

        # 2. Apply model transform
        model_inputs = self.model_transform({"messages": messages})

        # 3. Collate to batch size of 1 and tensor-ify
        if is_multimodal_input:
            batch = padded_collate_tiled_images_and_mask(
                [model_inputs], pad_direction="left"
            )
        else:
            batch = {"tokens": left_pad_sequence(
                [torch.tensor(model_inputs["tokens"])],
                batch_first=True,
            )}
        batch_to_device(batch, self._device)
        
        # 4. Prefill step
        generated_tokens = []
        t0 = time.perf_counter()
        logits = self.model(**batch)[:, -1]
        token = sample(logits, cfg.temperature, cfg.top_k)
        generated_tokens.append(token.item())

        if is_multimodal_input:
            cache_mask = {"encoder_mask": batch["encoder_mask"][:, -1:]}
        else:
            cache_mask = {}
        
        # 5. Continue generating
        for _ in range(cfg.max_new_tokens):
            if token.item() in self.model_transform.stop_tokens:
                break
            logits = self.model(token, **cache_mask)[:, -1]
            token = sample(logits, cfg.temperature, cfg.top_k)
            generated_tokens.append(token.item())
        t = time.perf_counter() - t0

        # 6. Decode tokens
        decoded = self.model_transform.decode(generated_tokens)
        logger.info(decoded)

        # 7. Log metrics
        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(
                    self.model.parameters(), self.model.buffers()
                )
            ]
        )
        tokens_generated = len(generated_tokens)
        tokens_sec = tokens_generated / t
        logger.info(
            f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
