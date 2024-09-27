# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any, Dict, List

import torch
from omegaconf import DictConfig, OmegaConf

from torchtune import config, training, utils
from torchtune.data import load_image, Message, padded_collate_tiled_images_and_mask

from torchtune.generation import sample

from torchtune.modules.transforms import Transform


class SingleTurnYAMLToMessages(Transform):
    """
    Converts a single turn conversation in YAML format to a list of messages.

    Expects the YAML to look like:
        system: You are a helpful AI assistant.
        user: What is the capital of France?

    or if it includes an image:
        system: You are a helpful AI assistant.
        user:
            image: url or path_to_image
            text: Describe the image in detail.
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

        # Finally, add an empty assistant message to kick-start generation
        messages.append(Message(role="assistant", content=""))
        return messages


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.
    This works for text-only generation and image-text generation.

    This *does not* currently support the following features:
        - torch.compile
        - quantization through torchao
        - multi-GPU generation
        - batch generation
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._logger = utils.get_logger(cfg.log_level)
        training.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        """Setup the model and transforms."""
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
        self.model_transform = config.instantiate(cfg.tokenizer)
        self.to_messages = SingleTurnYAMLToMessages()

    def log_metrics(self, total_time: int, tokens_per_second: float) -> None:
        """Logs the following metrics: total time for inference, tokens/sec,
        bandwidth achieved, and max memory allocated.

        Feel free to modify this function to log additional metrics.
        """
        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(self.model.parameters(), self.model.buffers())
            ]
        )
        self._logger.info(
            f"Time for inference: {total_time:.02f} sec total, {tokens_per_second:.02f} tokens/sec"
        )
        self._logger.info(
            f"Bandwidth achieved: {model_size * tokens_per_second / 1e9:.02f} GB/s"
        )
        self._logger.info(
            f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB"
        )

    @torch.inference_mode()
    def generate(self, cfg: DictConfig):
        """The main entry point for generating tokens from a prompt."""
        # 1. Convert input to messages
        messages = self.to_messages(OmegaConf.to_container(cfg.prompt))
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
                encoder_max_seq_len=(
                    self.model_transform.image_seq_len if is_multimodal_input else None
                ),
                decoder_max_seq_len=total_response_length,
            )

        # 4. Pre-allocate causal mask and input_pos
        causal_mask = torch.tril(
            torch.ones(
                size=(total_response_length, total_response_length),
                dtype=torch.bool,
                device=self._device,
            )
        )
        input_pos = torch.arange(total_response_length)

        # 5. Collate to batch size of 1 and tensor-ify
        batch = {}
        if is_multimodal_input:
            batch = padded_collate_tiled_images_and_mask(
                [model_inputs], pad_direction="left", pad_max_images=1
            )
            batch["encoder_mask"] = batch["encoder_mask"][:, :seq_len]
            prompt = batch.pop("tokens").to(self._device)
        else:
            prompt = torch.tensor(
                model_inputs["tokens"], device=self._device
            ).unsqueeze(0)
        batch["mask"] = causal_mask[None, :seq_len]
        batch["input_pos"] = input_pos[None, :seq_len]
        utils.batch_to_device(batch, self._device)

        # 6. Prefill step
        generated_tokens = []
        t0 = time.perf_counter()
        logits = self.model(prompt, **batch)[:, -1]
        token = sample(logits, temperature=cfg.temperature, top_k=cfg.top_k)
        generated_tokens.append(token.item())

        if is_multimodal_input:
            # Don't need image info b/c we only support 1 image and it's been
            # processed by the model now
            batch.pop("encoder_input")
            batch["encoder_mask"] = batch["encoder_mask"][:, -1:]

        # 7. Continue generating
        for i in range(cfg.max_new_tokens):

            # Update position and mask for incremental decoding
            batch["input_pos"] = input_pos[None, seq_len]
            batch["mask"] = causal_mask[None, seq_len, None, :]

            if token.item() in self.model_transform.stop_tokens:
                break

            logits = self.model(token, **batch)[:, -1]
            token = sample(logits, temperature=cfg.temperature, top_k=cfg.top_k)
            generated_tokens.append(token.item())
            seq_len += 1

        t = time.perf_counter() - t0

        # 8. Translate tokens back to text
        decoded = self.model_transform.decode(generated_tokens)
        self._logger.info(f"\n\n{decoded}\n")

        # 9. Log metrics
        tokens_per_second = len(generated_tokens) / t
        self.log_metrics(total_time=t, tokens_per_second=tokens_per_second)


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
