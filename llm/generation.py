@torch.no_grad()
def generate(
    decoder_lm: torch.nn.Module,
    prompt_tokens: List[List[int]],
    min_gen_len: int,
    max_gen_len: int,
    eos_token_id: int,
    pad_token_id: int,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 0,
    keep_prompt: bool = True,
    logprobs: bool = False,
    decoder_lm_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
    logits_accessor: Optional[Callable] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Interface for generation supporting temperature, top-k, and top-p sampling.

    Args:
        decoder_lm: Input module with which to run `forward`.
        prompt_tokens: List of tokenized per-batch prompts.
        min_gen_len: Minimum generated sequence length.
        max_gen_len: Maximum generated sequence length.
        eos_token_id: ID for end-of-sentence token.
        pad_token_id: ID for padding token.
        temperature: Temperature value to control sampling randomness.
        top_p: Probability threshold for nucleus sampling.
        top_k: Number of tokens kept for top-k filtering.
        keep_prompt: Whether to keep prompt tokens in the output tensor(s).
        logprobs: Whether to compute log probabilities.
        decoder_lm_kwargs: Additional arguments to pass to `decoder_lm.forward`.
        device: Device on which to initialize prompt token tensors (should match device of model).
        logits_accessor: Function to extract logits from model output.

    Returns:
        Tuple of generated tokens and optional log probabilities if `logprobs=True`,
        where the dimensions of each tensor are (batch_size, max_gen_length)

    Example:
    ```python
    >>> from transformers import AutoTokenizer, BertModel
    >>> from torchmultimodal.utils.text import generate

    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> model = BertModel.from_pretrained('bert-base-uncased')
    >>> input_prompt = "Today is a good day for"
    >>> prompt_tokens = [tokenizer.encode(input_prompt)]
    >>> tokens, token_logprobs = generate(
    ...     model,
    ...     prompt_tokens,
    ...     min_gen_len=1,
    ...     max_gen_len=10,
    ...     eos_token_id=tokenizer.eos_token_id,
    ...     pad_token_id=tokenizer.sep_token_id,
    ...     logprobs=True
    ... )
    ```
    """
    decoder_lm_kwargs = decoder_lm_kwargs or {}
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
    logits_transforms = _get_logits_transforms(temperature, top_p, top_k)
    # TODO: generalize the LLM's behavior - for example, models may not take in
    # a start_pos.
    prev_pos = 0
    eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for cur_pos in range(min_prompt_len, total_gen_len):
        input_ids = tokens[:, prev_pos:cur_pos]
        outputs = decoder_lm(input_ids, prev_pos)
        if logits_accessor:
            logits = logits_accessor(outputs)
        else:
            logits = outputs
        next_token_logits = logits[:, -1]

        # Convert to probability distribution, then sample
        next_token_probs = next_token_logits.softmax(dim=-1)
        next_token_probs = _apply_logits_transforms(logits_transforms, next_token_probs)
        next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)
        # Record positions of any EOS tokens across batches
        eos_reached_cur = next_token.eq(eos_token_id)
        eos_reached |= eos_reached_cur
        # Avoid overwriting the prompt for prompts that are longer than min_prompt_len.
        tokens[:, cur_pos] = torch.where(
            prompt_mask[:, cur_pos],
            tokens[:, cur_pos],
            # tokens[:, cur_pos].masked_scatter(~eos_reached, next_token),
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
        prev_pos = cur_pos
        if eos_reached.all().item():
            break

    if not keep_prompt:
        tokens = tokens[:, max_prompt_len:]
        if token_logprobs is not None:
            token_logprobs = token_logprobs[:, max_prompt_len:]

    return tokens, token_logprobs if logprobs else None
