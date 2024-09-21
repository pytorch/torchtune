# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any, Dict, List, Union

import torch
from omegaconf import DictConfig

from torchtune import config, training, utils
from torchtune.data import (
    left_pad_sequence,
    load_image,
    Message,
    padded_collate_tiled_images_and_mask,
)

from torchtune.generation import (
    sample,
)

from torchtune.modules.transforms import Transform


class SingleTurnYAMLToMessages(Transform):
    """
    Converts a single turn conversation in YAML format to a list of messages.
    """

    def __call__(self, prompt: Dict[str, Any]) -> List[Message]:
        messages = []

        # Iterate through roles and add content
        for role, content in prompt.items():
            if isinstance(content, str):
                new_content = [{"type": "text", "content": content}]
            else:
                assert (
                    "image" in content.keys()
                ), "Multiple entries per role expect an image key"
                image_loc = content["image"]
                image = load_image(image_loc)
                new_content = [
                    {"type": "image", "content": image},
                    {"type": "text", "content": content["text"]},
                ]
            messages.append(Message(role=role, content=new_content))

        # Finally, add an empty assistant message
        messages.append(
            Message(role="assistant", content=[{"type": "text", "content": ""}])
        )
        return messages


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.
    This works for text-only generation and image-text generation.

    Warning:
        This has only been extensively tested with the following configs:
            - multimodal_generation

    This *does not* currently support the following features:
        - torch.compile
        - quantization through torchao
        - speculative decoding
        - multi-GPU generation
        - batch generation
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._logger = utils.get_logger(cfg.log_level)
        training.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        # Load checkpointer and state_dict
        _checkpointer = config.instantiate(cfg.checkpointer)
        _ckpt_dict = _checkpointer.load_checkpoint()

        # Instantiate model
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg.model)
        model.load_state_dict(_ckpt_dict[training.MODEL_KEY])
        self.model = model
        self._logger.info(f"Model was initialized with precision {self._dtype}.")

        # Instantiate transforms
        self.model_transform = config.instantiate(cfg.transform)
        self.to_messages = SingleTurnYAMLToMessages()

    @torch.inference_mode()
    def generate(self, cfg: DictConfig):
        # 1. Convert input to messages
        messages = self.to_messages(cfg.prompt)
        is_multimodal_input = any([m.contains_media for m in messages])

        # 2. Apply model transform
        model_inputs = self.model_transform({"messages": messages}, inference=True)
        seq_len = len(model_inputs["tokens"])
        total_response_length = seq_len + cfg.max_new_tokens

        # 3. Setup KV cache
        with self._device:
            self.model.setup_caches(
                batch_size=1,
                dtype=self._dtype,
                decoder_max_seq_len=total_response_length,
            )

        # 4. Setup masks and input_pos
        causal_mask = torch.tril(
            torch.ones(
                total_response_length,
                total_response_length,
                dtype=torch.bool,
                device=self._device,
            )
        ).unsqueeze(0)
        input_pos = torch.arange(total_response_length).unsqueeze(0)

        # 5. Collate to batch size of 1 and tensor-ify
        if is_multimodal_input:
            batch = padded_collate_tiled_images_and_mask(
                [model_inputs], pad_direction="left"
            )
        else:
            batch = {
                "tokens": torch.tensor(model_inputs["tokens"]).unsqueeze(0),
                "mask": causal_mask[:, :seq_len],
                "input_pos": input_pos[:, :seq_len],
            }
        utils.batch_to_device(batch, self._device)

        # 6. Prefill step
        generated_tokens = []
        t0 = time.perf_counter()
        logits = self.model(**batch)[:, -1]
        token = sample(logits, temperature=cfg.temperature, top_k=cfg.top_k)
        generated_tokens.append(token.item())

        # 7. Continue generating
        for i in range(cfg.max_new_tokens):
            # Update batch params
            batch_params = {
                "input_pos": input_pos[:, seq_len],
                "mask": causal_mask[:, seq_len, None, :],
                "encoder_mask": (
                    batch["encoder_mask"][:, -1:] if is_multimodal_input else None
                ),
            }

            if token.item() in self.model_transform.stop_tokens:
                break

            logits = self.model(token, **batch_params)[:, -1]
            token = sample(logits, temperature=cfg.temperature, top_k=cfg.top_k)
            generated_tokens.append(token.item())
            seq_len += 1

        t = time.perf_counter() - t0

        # 8. Decode tokens
        decoded = self.model_transform.decode(generated_tokens)
        self._logger.info(decoded)

        # 9. Log metrics
        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(self.model.parameters(), self.model.buffers())
            ]
        )
        tokens_sec = len(generated_tokens) / t
        self._logger.info(
            f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        self._logger.info(
            f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
        )
        self._logger.info(
            f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB"
        )


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
