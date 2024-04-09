# Run via torchrun --nproc_per_node 2 test_gemma_share_weights.py
import torch
import sys
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed import init_process_group, destroy_process_group
from torchtune import modules, utils
from typing import Optional
import torch.nn as nn
from torch import Tensor
from torchtune.modules import KVCache
from torchtune.models.gemma import gemma
from torchtune.models.gemma._component_builders import gemma_mlp
from torchtune.modules.transformer import _get_clones, TransformerDecoderLayer
from torchtune.utils import FullModelHFCheckpointer, ModelType
from torchtune.modules import (
    CausalSelfAttention,
    KVCache,
    RotaryPositionalEmbeddings,
    TransformerDecoderLayer,
)
from torchtune.models.gemma.rms_norm import GemmaRMSNorm

log = utils.get_logger("DEBUG")

MODIFIED = False

class GemmaTransformerDecoderModified(nn.Module):
    def __init__(
        self,
        tok_embeddings: nn.Embedding,
        layer: TransformerDecoderLayer,
        num_layers: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        norm_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.tok_embeddings = tok_embeddings
        self.layers = _get_clones(layer, num_layers)
        self.norm = norm
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal_mask = None
        self.norm_embeddings = norm_embeddings

    def setup_caches(self, max_batch_size: int, dtype: torch.dtype) -> None:
        for layer in self.layers:
            layer.attn.kv_cache = KVCache(
                max_batch_size=max_batch_size,
                max_seq_len=self.max_seq_len,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )

        # causal_mask is used during inference to ensure we're attending
        # to the right tokens
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool)
        )

    def forward(self, tokens: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        mask = None
        if self.causal_mask is not None:
            if input_pos is None:
                raise ValueError(
                    "Caches are setup, but the position of input token is missing"
                )
            # shape: [1, input_pos_len, m_s]
            # in most cases input_pos_len should be 1
            mask = self.causal_mask[None, None, input_pos]

        if self.norm_embeddings:
            hidden_dim = h.size(-1)
            h = h * torch.tensor(hidden_dim**0.5, dtype=h.dtype)

        for layer in self.layers:
            # shape: [b, s, d]
            h = layer(h, mask, None)

        # shape: [b, s, d]
        h = self.norm(h)

        # shape: [b, s, v]
        output = self.tok_embeddings(h).float()
        return output

def gemma_modified(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-6,
    rope_base: int = 10_000,
    norm_embeddings: bool = True,
) -> GemmaTransformerDecoderModified:
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    self_att = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
        k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
        pos_embeddings=rope,
        kv_cache=None,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    mlp = gemma_mlp(dim=embed_dim, hidden_dim=intermediate_dim)
    layer = TransformerDecoderLayer(
        attn=self_att,
        mlp=mlp,
        sa_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
        mlp_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
    )
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    model = GemmaTransformerDecoderModified(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
        norm_embeddings=norm_embeddings,
    )
    return model

def fsdp_wrap_original(model: nn.Module, device: torch.device, rank: int):
    return FSDP(
        module=model,
        auto_wrap_policy=ModuleWrapPolicy({modules.TransformerDecoderLayer}),
        sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
        device_id=device,
        # this recipe does not currently support mixed precision training
        mixed_precision=None,
        # Ensure we broadcast params and buffers from rank 0
        sync_module_states=True,
        # Initialize empty modules on all non-zero ranks
        param_init_fn=(
            lambda module: module.to_empty(
                device=torch.device("cuda"), recurse=False
            )
            if rank != 0
            else None
        ),
    )

def fsdp_wrap_gemma(model: nn.Module, device: torch.device, rank: int):
    return FSDP(
        module=model,
        auto_wrap_policy=ModuleWrapPolicy({modules.TransformerDecoderLayer}),
        sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
        device_id=device,
        # this recipe does not currently support mixed precision training
        mixed_precision=None,
        # Ensure we broadcast params and buffers from rank 0
        sync_module_states=False,
        # Initialize empty modules on all non-zero ranks
        param_init_fn=None,
    )


def main() -> None:
    init_process_group("nccl")
    device = utils.get_device(device='cuda')
    _, rank = utils.get_world_size_and_rank()
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir="/tmp/gemma/",
        checkpoint_files=[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        recipe_checkpoint=None,
        output_dir="/tmp/gemma",
        model_type=ModelType.GEMMA,
    )
    sd = checkpointer.load_checkpoint()[utils.MODEL_KEY]
    log.info("done loading checkpoint")
    log.info(sd.keys())
    if rank == 0:
        with utils.set_default_dtype(torch.bfloat16):
            if MODIFIED:
                model = gemma_modified(
                    vocab_size=256_000,
                    num_layers=18,
                    num_heads=8,
                    head_dim=256,
                    num_kv_heads=1,
                    embed_dim=2048,
                    intermediate_dim=16384,
                    max_seq_len=8192,
                    attn_dropout=0.0,
                    norm_eps=1e-6,
                )
            else:
                model = gemma(
                    vocab_size=256_000,
                    num_layers=18,
                    num_heads=8,
                    head_dim=256,
                    num_kv_heads=1,
                    embed_dim=2048,
                    intermediate_dim=16384,
                    max_seq_len=8192,
                    attn_dropout=0.0,
                    norm_eps=1e-6,
                )
                sd["output.weight"] = sd["tok_embeddings.weight"]
        log.info("done with model instantiation")
        model.load_state_dict(sd)
        log.info("done loading state dict on rank zero")
        log.info(model.state_dict().keys())
    else:
        # For non-zero ranks, load the model on meta device
        with utils.set_default_dtype(torch.bfloat16), torch.device("meta"):
            if MODIFIED:
                model = gemma_modified(
                    vocab_size=256_000,
                    num_layers=18,
                    num_heads=8,
                    head_dim=256,
                    num_kv_heads=1,
                    embed_dim=2048,
                    intermediate_dim=16384,
                    max_seq_len=8192,
                    attn_dropout=0.0,
                    norm_eps=1e-6,
                )
            else:
                model = gemma(
                    vocab_size=256_000,
                    num_layers=18,
                    num_heads=8,
                    head_dim=256,
                    num_kv_heads=1,
                    embed_dim=2048,
                    intermediate_dim=16384,
                    max_seq_len=8192,
                    attn_dropout=0.0,
                    norm_eps=1e-6,
                )
    model = model.to(torch.bfloat16)

    log.info("start fsdp wrapping")

    # Wrap the model with FSDP. This will ensure that the model is sharded
    # across all available GPUs.
    model = fsdp_wrap_original(model, device, rank)
    # with FSDP.summon_full_params(model):
    #     model.output.weight = model.tok_embeddings.weight
    log.info("done fsdp wrapping")
    log.info(model.state_dict().keys())
    torch.distributed.barrier()
    log.info("after barrier")
    inputs = torch.randint(0, 32_000, (2, 100))
    out = model(inputs)
    log.info(out.shape, out.sum())
    destroy_process_group()

if __name__ == "__main__":
    sys.exit(main())
