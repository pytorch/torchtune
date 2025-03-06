from typing import Optional

import torch
from torchtune import config, generation, modules, rlhf, training, utils

from torchtune.dev.grpo.generation import generate
from torchtune.dev.rl.generators.base import (
    GeneratorABC,
    GeneratorInput,
    GeneratorOutput,
)
from torchtune.modules import local_kv_cache


class TorchTuneGenerator(GeneratorABC):
    def __init__(
        self,
        tokenizer,
        model,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        context_length: int,
        max_generated_tokens: int,
        num_generations_per_prompt: int,
        temperature: float = 1.0,
        top_k: int = 1,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.max_generated_tokens = max_generated_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.rng = rng

        self.batch_size, self.context_length, self.num_generations_per_prompt = (
            batch_size,
            context_length,
            num_generations_per_prompt,
        )

    def generate(self, input: GeneratorInput) -> GeneratorOutput:
        tokens = input.tokens

        batch_input_ids = tokens[:, None, :].expand(
            -1, self.num_generations_per_prompt, -1
        )  # [B, G, L]
        batch_input_ids = batch_input_ids.reshape(
            self.batch_size * self.num_generations_per_prompt, -1
        )

        # step 1: generate responses, and logits corresponding to the responses using the current policy
        with local_kv_cache(
            model=self.model,
            batch_size=self.batch_size * self.num_generations_per_prompt,
            device=self.device,
            dtype=self.dtype,
            decoder_max_seq_len=self.context_length + self.max_generated_tokens,
        ):
            query_responses, logits = generate(  # [B x G, L], [B x G, L, V]
                model=self.model,
                prompt=batch_input_ids,
                max_generated_tokens=self.max_generated_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                pad_id=self.tokenizer.pad_id,
                rng=self.rng,
                stop_tokens=self.tokenizer.stop_tokens,
                return_logits=True,
            )
            assert logits is not None

        # We had a barrier and a reshard here in the original code, before we offloaded the generator to a separate ray worker
        # torch.distributed.barrier()
        # # Use training.distributed instead of training._distributed
        # training.distributed.recursive_reshard(self.model)
        # torch.cuda.empty_cache()  <--- I also don't think we need this?

        responses = query_responses[:, self.context_length :].clone()
        query_response_padding_masks = query_responses != self.tokenizer.pad_id

        # step 1.1 create attention masks and position IDs for any padding tokens in inputs, used for future forward passes
        masks = generation.get_causal_mask_from_padding_mask(
            query_response_padding_masks
        )
        position_ids = generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )
        del query_response_padding_masks

        # The original code regenerated the logits here, but in our case we want them to be the same as the ones we generated above
        logits = logits[:, self.context_length - 1 :]
        logprobs = rlhf.batched_logits_to_logprobs(logits, responses, self.temperature)
        del logits
        torch.cuda.empty_cache()  # Keeping this for now, but flushing cache seems expensive

        # Convert tuple to GeneratorOutput
        return GeneratorOutput(
            query_responses=query_responses,
            responses=responses,
            logprobs=logprobs,
            masks=masks,
            position_ids=position_ids,
        )
