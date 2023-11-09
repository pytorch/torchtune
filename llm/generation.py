from torch import nn, Tensor
import torch
from typing import List, Optional, Dict

from llm.utils.logits_transforms import LogitsTransform, TopKTransform, TopPTransform, TemperatureTransform

class GenerationUtils:
    """Utility class for generating text from an LLaMA model.

    Args:
        decoder_lm: Transformer-Decoder based language model.
        pad_id: Padding token ID.
        eos_id: End-of-sequence token ID.
    """
    def __init__(self, decoder_lm: nn.Module, pad_id: int, eos_id: int):
        self.decoder_lm = decoder_lm
        self.pad_id = pad_id
        self.eos_id = eos_id

    def _get_logits_transforms(
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> List[LogitsTransform]:
        """Returns a list of parameterized logits transforms that can be chained."""
        logits_transforms = []
        if temperature > 0:
            logits_transforms.append(TemperatureTransform(temperature))
        if top_p > 0:
            logits_transforms.append(TopPTransform(top_p))
        if top_k > 1:
            logits_transforms.append(TopKTransform(top_k))

        return logits_transforms


    def _apply_logits_transforms(
        logits_transforms: List[LogitsTransform], logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Applies a chained list of logits transforms."""
        output_logits = (
            functools.reduce(lambda x, f: f(x), logits_transforms, logits)
            if logits_transforms
            else logits
        )
        return output_logits

    @torch.no_grad()
    def generate(
        prompt_tokens: List[List[int]],
        min_gen_len: int,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 1,
        keep_prompt: bool = True,
        logprobs: bool = False,
        decoder_lm_kwargs: Dict[str, Any] = {},
        decode_incrementally: bool = True,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Interface for generation supporting temperature, top-k, and top-p sampling.

        Args:
            prompt_tokens: List of tokenized per-batch prompts.
            min_gen_len: Minimum generated sequence length.
            max_gen_len: Maximum generated sequence length.
            temperature: Temperature value to control sampling randomness.
            top_p: Probability threshold for nucleus sampling.
            top_k: Number of tokens kept for top-k filtering.
            keep_prompt: Whether to keep prompt tokens in the output tensor(s).
            logprobs: Whether to compute log probabilities.
            decoder_lm_kwargs: Additional arguments to pass to `decoder_lm.forward`.
            device: Device on which to initialize prompt token tensors (should match device of model).

        Returns:
            Tuple of generated tokens and optional log probabilities if `logprobs=True`,
            where the dimensions of each tensor are (batch_size, max_gen_length)

        Example:
            >>> LLaMA = GenerationUtils(model, pad_id = tokenizer.pad_id, eos_id = tokenizer.eos_id)
            >>> tokens = LLaMA.generate(
            ...     [tokenizer.encode(["I love to eat"])],
            ...     min_gen_len=5,
            ...     max_gen_len=20,
            ...     temperature=0.8,
            ...     top_p=0.7,
            ...     keep_prompt=True,
            ... )
            >>> print(tokens)
            ["I love to eat ice cream"]
        """
        batch_size = len(prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        min_prompt_len = min(len(p) for p in prompt_tokens)
        total_gen_len = max_gen_len + max_prompt_len
        tokens = torch.full(
            (batch_size, total_gen_len), pad_token_id, dtype=torch.long, device=device
        )
        for i, prompt in enumerate(prompt_tokens):
            tokens[i, : len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=device)
        if logprobs:
            token_logprobs = torch.full_like(
                tokens, float("-inf"), dtype=torch.float, device=device
            )
        else:
            token_logprobs = None
        # mask to ensure we don't overwrite the prompt for prompts > min_prompt_len.
        prompt_mask = tokens != pad_token_id
        logits_transforms = self._get_logits_transforms(temperature, top_p, top_k)
        # TODO: generalize the LLM's behavior - for example, models may not take in
        # a start_pos.
        prev_pos = 0
        eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for cur_pos in range(min_prompt_len, total_gen_len):
            input_ids = tokens[:, prev_pos:cur_pos]
            logits = self.decoder_lm(input_ids, prev_pos, **decoder_lm_kwargs)
            next_token_logits = logits[:, -1]

            # Convert to probability distribution, then sample
            next_token_probs = next_token_logits.softmax(dim=-1)
            next_token_probs = self._apply_logits_transforms(logits_transforms, next_token_probs)
            next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)
            # Record positions of any EOS tokens across batches
            eos_reached_cur = next_token.eq(eos_token_id)
            eos_reached |= eos_reached_cur
            # Avoid overwriting the prompt for prompts that are longer than min_prompt_len.
            tokens[:, cur_pos] = torch.where(
                prompt_mask[:, cur_pos],
                tokens[:, cur_pos],
                next_token,
            )
            if token_logprobs is not None:
                token_logprobs[:, cur_pos].masked_scatter_(
                    ~eos_reached,
                    -F.cross_entropy(
                        next_token_logits,
                        tokens[:, cur_pos],
                        reduction="none",
                        ignore_index=pad_token_id,
                    ),
                )

            if decode_incrementally:
                prev_pos = cur_pos

            if eos_reached.all().item():
                break

        if not keep_prompt:
            tokens = tokens[:, max_prompt_len:]
            if token_logprobs is not None:
                token_logprobs = token_logprobs[:, max_prompt_len:]

        return tokens, token_logprobs if logprobs else None
