# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.attention.flex_attention import create_block_mask

from torchtune import config, training, utils
from torchtune.data import (
    load_image,
    Message,
    padded_collate,
    padded_collate_tiled_images_and_mask,
)

from torchtune.generation import sample
from torchtune.modules.attention_utils import causal_mask_flex, kv_offset_mask_flex
from torchtune.modules.model_fusion import DeepFusionModel, EarlyFusionModel

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

    def __call__(self, prompt: dict[str, Any]) -> list[Message]:
        messages = []

        # Iterate through roles and add content
        for role, content in prompt.items():
            if content is None:
                continue
            elif isinstance(content, str):
                new_content = [{"type": "text", "content": content}]
            elif "image" in content.keys():
                image_loc = content["image"]
                image = load_image(image_loc)
                new_content = [
                    {"type": "image", "content": image},
                    {"type": "text", "content": content["text"]},
                ]
            else:
                assert (
                    "text" in content.keys()
                ), "Multiple entries per role expect at least a text key"
                new_content = [{"type": "text", "content": content["text"]}]
            messages.append(Message(role=role, content=new_content))

        # Finally, add an empty assistant message to kick-start generation
        messages.append(Message(role="assistant", content=""))
        return messages


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.
    This works for text-only generation and image-text generation.

    Supports distributed inference using Tensor Paralellism(TP) for
    large models that don't fit on a single GPU. For more information
    on TP, see: https://pytorch.org/docs/stable/distributed.tensor.parallel.html.

    This *does not* currently support the following features:
        - torch.compile
        - quantization through torchao
        - batch generation
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._logger = utils.get_logger(cfg.log_level)
        # Set up distributed env
        dist.init_process_group(backend="nccl")
        _, rank = utils.get_world_size_and_rank()
        self._is_rank_zero = rank == 0
        training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )

    def setup(self, cfg: DictConfig) -> None:
        """Setup the model and transforms."""
        # Load checkpointer and state_dict
        _checkpointer = config.instantiate(cfg.checkpointer)
        _ckpt_dict = _checkpointer.load_checkpoint()

        # Instantiate model on meta device
        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg.model)

        # Set up tensor parallel device mesh
        tp_degree = dist.get_world_size()  # Using all GPUs for TP
        tp_mesh_shape = (tp_degree,)
        tp_device_mesh = dist.init_device_mesh("cuda", tp_mesh_shape)

        # Use the local number (num_heads, num_kv_heads, embed_dim) to account for tensor paralell
        model = training.prepare_mha_for_tp(model, tp_device_mesh)
        parallelize_module(
            model,
            tp_device_mesh,
            parallelize_plan=config.instantiate(
                cfg.tensor_parallel_plan, model=model, inference=True
            ),
        )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model=model,
            full_sd=_ckpt_dict[training.MODEL_KEY],
            device=self._device,
            strict=True,
            cpu_offload=False,
            use_distributed_state_dict=cfg.get("use_distributed_state_dict", False),
        )

        self.model = model
        self.model.eval()
        if self._is_rank_zero:
            config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
            self._logger.info(
                f"Model was initialized with precision {self._dtype} and TP degree {tp_degree}."
            )

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
            f"Bandwidth achieved: {model_size * tokens_per_second / (1024**3):.02f} GiB/s"
        )
        if self._device.type != "cpu":
            torch_device = utils.get_torch_device_namespace()
            self._logger.info(
                f"Max memory allocated: {torch_device.max_memory_allocated() / (1024**3):.02f} GiB"
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
        if isinstance(self.model, DeepFusionModel) and is_multimodal_input:
            batch = padded_collate_tiled_images_and_mask(
                [model_inputs],
                pad_direction="left",
                pad_max_images=1,
                pad_max_tiles=self.model_transform.max_num_tiles,
            )
            batch["encoder_mask"] = batch["encoder_mask"][:, :seq_len]
            prompt = batch.pop("tokens").to(self._device)
        elif isinstance(self.model, EarlyFusionModel) and is_multimodal_input:
            batch = padded_collate(
                [model_inputs],
                pad_direction="left",
                keys_to_pad=["tokens"],
                padding_idx=self.model_transform.pad_id,
            )
            prompt = batch.pop("tokens").to(self._device)
        else:
            prompt = torch.tensor(
                model_inputs["tokens"], device=self._device
            ).unsqueeze(0)
        use_flex = cfg.get("use_flex", False)
        if use_flex:
            batch["mask"] = create_block_mask(
                causal_mask_flex,
                1,
                None,
                seq_len,
                total_response_length,
                device="cuda",
            )
        else:
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
            if "encoder_mask" in batch:
                batch["encoder_mask"] = batch["encoder_mask"][:, -1:]

        # 7. Continue generating
        for i in range(cfg.max_new_tokens):
            # Update position and mask for incremental decoding
            batch["input_pos"] = input_pos[None, seq_len]
            if use_flex:
                batch["mask"] = create_block_mask(
                    partial(kv_offset_mask_flex, offset=seq_len),
                    1,
                    None,
                    1,
                    total_response_length,
                    device="cuda",
                )
            else:
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
        if self._is_rank_zero:
            self._logger.info(f"\n\n{decoded}\n")

        # 9. Log metrics
        tokens_per_second = len(generated_tokens) / t
        if self._is_rank_zero:
            self.log_metrics(total_time=t, tokens_per_second=tokens_per_second)


@config.parse
def main(cfg: DictConfig) -> None:
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
