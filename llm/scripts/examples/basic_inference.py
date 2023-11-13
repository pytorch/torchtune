# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import argparse

import functools

import logging

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from llm.llama2.tokenizer import Tokenizer
from llm.llama2.transformer import TransformerDecoder
from tests.llm.llama2.scripts.compare_decoder import Transformer
from transformers import LlamaForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


class MMRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x) -> torch.FloatTensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert ndim > 1
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    To be replaced by torchMM's RotaryEmbedding
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# TODO: below are taken from facebookresearch/llama, update to use torchMM components
# once they are compatible
class Attention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        dim: int,
        max_batch_size: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(
            dim,
            self.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self._init_cache_k()
        self._init_cache_v()

    def _init_cache_k(self):
        self.cache_k = torch.zeros(
            (
                self.max_batch_size,
                self.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def _init_cache_v(self):
        self.cache_v = torch.zeros(
            (
                self.max_batch_size,
                self.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        # kv-cache for inference only, disabled during training.
        use_cache = not self.training

        if use_cache:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, seqlen + start_pos, n_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, seqlen + start_pos, n_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        import math

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_heads, seqlen, cache_len + seqlen)
        import torch.nn.functional as F

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        hidden_layer_dim_multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # round hidden dimension to `multiple_of`
        hidden_dim = hidden_layer_dim_multiple_of * (
            (hidden_dim + hidden_layer_dim_multiple_of - 1)
            // hidden_layer_dim_multiple_of
        )

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MMTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        n_heads: int,
        n_kv_heads: int,
        dim: int,
        hidden_layer_dim_multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        max_batch_size: int,
        max_seq_len: int,
        norm_eps: float,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(
            n_heads, n_kv_heads, dim, max_batch_size, max_seq_len
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            hidden_layer_dim_multiple_of=hidden_layer_dim_multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = MMRMSNorm(dim, eps=norm_eps)
        self.ffn_norm = MMRMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class MMTransformer(nn.Module):
    """
    LLama2 implementation, free of any coupling to parallelism implementations, heavily drawn from
    https://github.com/facebookresearch/llama.
    """

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        hidden_layer_dim_multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        max_batch_size: int,
        max_seq_len: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dim = dim
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.layers = torch.nn.ModuleList()
        for layer_id in range(n_layers):
            self.layers.append(
                MMTransformerBlock(
                    layer_id,
                    n_heads,
                    n_kv_heads,
                    dim,
                    hidden_layer_dim_multiple_of,
                    ffn_dim_multiplier,
                    max_batch_size,
                    max_seq_len,
                    norm_eps,
                )
            )

        self.norm = MMRMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.dim // self.n_heads, self.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):

        # truncate the input tokens to fit into the max sequence length,
        # and pull start position into the correct range
        start_pos = start_pos % self.max_seq_len
        tokens = tokens[:, : self.max_seq_len]

        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output


class LogitsTransform(abc.ABC):
    """Interface for a logits transformation."""

    @abc.abstractmethod
    def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        pass


class TemperatureTransform(LogitsTransform):
    """Controls randomness of predicted tokens via a temperature value.

    Args:
        temperature (float): The parameter controlling distribution randomness.
    """

    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores /= self.temperature
        return scores


class TopPTransform(LogitsTransform):
    """Filters the distribution to cover the fewest tokens whose cumulative mass
    exceeds `prob`.

    Args:
        prob (float): The minimum cumulative probability mass that the kept tokens
            must cover.
    """

    def __init__(self, prob: float):
        self.prob = prob

    def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_sort, scores_index = torch.sort(scores, dim=-1, descending=True)
        scores_cumulative = scores_sort.cumsum(dim=-1)

        # Ignore tokens introducing more probability mass than needed
        discard_mask = scores_cumulative - scores_sort > self.prob
        scores_sort[discard_mask] = 0.0

        scores_sort.div_(scores_sort.sum(dim=-1, keepdim=True))  # renormalize
        scores.scatter_(-1, scores_index, scores_sort)
        return scores


class TopKTransform(LogitsTransform):
    """Filters the distribution to include the top-k highest probability tokens.

    Args:
        top_k (int): The number of highest probability tokens to keep.
    """

    def __init__(self, top_k: int):
        self.top_k = top_k

    def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))
        scores_topk, _ = scores.topk(top_k)

        discard_mask = scores < scores_topk[..., -1]
        scores.masked_fill_(discard_mask, 0.0)

        scores.div_(scores.sum(dim=-1, keepdim=True))  # renormalize
        return scores


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
    if top_k > 0:
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
        decoder_lm (torch.nn.Module): Input module with which to run `forward`.
        prompt_tokens (List[List[int]]): List of tokenized per-batch prompts.
        min_gen_len (int): Minimum generated sequence length.
        max_gen_len (int): Maximum generated sequence length.
        eos_token_id (int): ID for end-of-sentence token.
        pad_token_id (int): ID for padding token.
        temperature (float): Temperature value to control sampling randomness.
        top_p (float): Probability threshold for nucleus sampling.
        top_k (int): Number of tokens kept for top-k filtering.
        keep_prompt (bool): Whether to keep prompt tokens in the output tensor(s).
        logprobs (bool): Whether to compute log probabilities.
        decoder_lm_kwargs (Optional[Dict[str, Any]]): Additional arguments to pass to `decoder_lm.forward`.
        device (Optional[torch.device]): Device on which to initialize prompt token tensors (should match device of model).
        logits_accessor (Optional[Callable]): Function to extract logits from model output.

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
        input_ids = tokens[:, :cur_pos]
        outputs = (
            decoder_lm(input_ids, 0) if decoder_lm_kwargs else decoder_lm(input_ids)
        )
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


@dataclass
class LlamaArgs:
    """
    Dataclass encapsulating various args to instantiate a Llama-2 decoder. The defaults
    are those of a 7b parameter model with a max_seq_len of 2048.

    Args:
        vocab_size (int): Number of entries in vocabulary (default: 32_000)
        embed_dim: (int): Embedding dimension (default: 4096)
        num_layers: (int): Number of Transformer layers (default: 32)
        num_heads (int): Number of attention heads (per layer). (default: 32)
        num_kv_heads: (Optional[int]): Number of key and value heads. This needs to
            be < num_heads and num_heads % num_kv_heads must be 0. `num_kv_heads` can be
            modified to implement GQA or MHA. The default is `None`, in which case
            `num_kv_heads` is set to `num_heads` and MHA is used. Please see
            llm.llama2.attention.LlamaSelfAttention for details.
        max_seq_len: int: Maximum sequence length that this model accepts. Default: 2048
    """

    vocab_size: int = 32_000
    embed_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: Optional[int] = None
    max_seq_len: int = 2048


def args_7b() -> LlamaArgs:
    return LlamaArgs(
        vocab_size=32_000,
        embed_dim=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=None,
        max_seq_len=2048,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Example 7B native Llama-2 inference.
        """
    )
    parser.add_argument(
        "--native-checkpoint-path", type=str, help="Path to native checkpoint file."
    )
    parser.add_argument(
        "--fair-checkpoint-path", type=str, help="Path to FAIR checkpoint file"
    )
    parser.add_argument(
        "--mm-checkpoint-path", type=str, help="Path for torchMM checkpoints"
    )
    parser.add_argument("--tokenizer-path", type=str, help="Path to tokenization file.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="""
        Device to initialize tensors on. This defaults to cuda:0 which is much faster
        than CPU for checkpoint conversion. If GPU is unavailable, pass in "cpu".
        """,
    )
    args = parser.parse_args()
    torch.set_default_device(args.device)
    if "cuda" in args.device:
        torch.cuda.set_device(args.device)
    llama_7b_args = args_7b()
    # import pdb ; p
    decoder = TransformerDecoder(
        vocab_size=llama_7b_args.vocab_size,
        num_layers=llama_7b_args.num_layers,
        num_heads=llama_7b_args.num_heads,
        num_kv_heads=llama_7b_args.num_kv_heads,
        embed_dim=llama_7b_args.embed_dim,
        max_seq_len=llama_7b_args.max_seq_len,
        norm_eps=1e-5,
        max_bsz_for_kv_cache=None,
    ).cpu()

    native_state_dict = torch.load(args.native_checkpoint_path)
    missing, unexpected = decoder.load_state_dict(native_state_dict, strict=False)
    tformer = Transformer(
        vocab_size=llama_7b_args.vocab_size,
        dim=llama_7b_args.embed_dim,
        n_layers=llama_7b_args.num_layers,
        n_heads=llama_7b_args.num_heads,
        max_seq_len=llama_7b_args.max_seq_len,
        n_kv_heads=llama_7b_args.num_kv_heads,
    ).cpu()

    fair_state_dict = torch.load(args.fair_checkpoint_path)
    m2, u2 = tformer.load_state_dict(fair_state_dict, strict=False)
    del fair_state_dict
    mm_tformer = MMTransformer(
        vocab_size=llama_7b_args.vocab_size,
        n_layers=llama_7b_args.num_layers,
        dim=llama_7b_args.embed_dim,
        n_heads=llama_7b_args.num_heads,
        n_kv_heads=llama_7b_args.num_kv_heads,
        hidden_layer_dim_multiple_of=256,
        ffn_dim_multiplier=None,
        max_batch_size=2,
        max_seq_len=llama_7b_args.max_seq_len,
    ).cpu()
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint import FileSystemReader

    sd = mm_tformer.state_dict()
    dist_cp.load_state_dict(
        state_dict=sd,
        storage_reader=FileSystemReader(args.mm_checkpoint_path),
        no_dist=True,
    )
    me, une = mm_tformer.load_state_dict(sd, strict=False)
    # me, une = mm_tformer.load_state_dict(native_state_dict, strict=False)
    del native_state_dict
    decoder.eval()
    tformer.eval()
    mm_tformer.eval()
    print(f"RV: tbd missing unexpected {missing} {unexpected}")
    print(f"RV: fair missing & unexpected: {m2} {u2}")
    print(f"RV: mm missing unexpected: {me} {une}")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    auth_token = "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"
    model_hf = LlamaForCausalLM.from_pretrained(  # pyre-ignore[16]
        "meta-llama/Llama-2-7b-hf",
        use_auth_token=auth_token,
        token=None,
    )
    # Inference
    prompts = [
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]

    tokens = torch.tensor(tokenizer.encode(prompts[0], add_eos=False), dtype=torch.long)
    toks = [tokenizer.encode(prompt, add_eos=False) for prompt in prompts]
    import pdb

    pdb.set_trace()
    generations, _ = generate(
        decoder_lm=decoder,
        prompt_tokens=toks,
        min_gen_len=1,
        max_gen_len=64,
        top_p=0,
        top_k=0,
        temperature=1.0,
        eos_token_id=tokenizer.eos_id,
        pad_token_id=tokenizer.pad_id,
        device=torch.cuda.current_device(),
        decoder_lm_kwargs=True,
    )
