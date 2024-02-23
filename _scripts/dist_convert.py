# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
import os
from pathlib import Path
from typing import Dict, Union

import torch
import torch.distributed as dist
from convert_checkpoint import _convert_llama_from_fair

# For checkpoint reading directly from manifold
# import torch.manifold.patch
from torch import nn, Tensor
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed.fsdp._fsdp_extensions import (
    _ext_chunk_dtensor,
    _ext_chunk_tensor,
)

log = logging.getLogger(__name__)


def _verify_fqn_across_ranks(fqn, grp_gloo):
    olist = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(olist, fqn, group=grp_gloo)
    assert len(set(olist)) == 1
    assert olist[0] == fqn


def _all_gather_into_list(data_tensor, model_parallel_group):
    tensor_list = [
        torch.zeros_like(data_tensor).cuda()
        for _ in range(dist.get_world_size(model_parallel_group))
    ]
    dist.all_gather(tensor_list, data_tensor.cuda(), group=model_parallel_group)
    return tensor_list


def _is_tp_sharded(fqn: str) -> bool:
    """
    Returns whether a tensor given by the fqn is tensor parallel sharded.
    NOTE: this is currently done by inspection of the model and is quite
    brittle and would need to be updated if the sharding changes.
    """
    return (
        "attention" in fqn
        or "feed_forward" in fqn
        or "output" in fqn
        or "tok_embeddings" in fqn
    )


def _unshard_param(
    ref_state_dict,
    fqn,
    model_parallel_group,
    grp_gloo,
    data_tensor,
    tp_sharded_shape,
):
    """
    Unshards the row or col-wise sharded parameter.
    For rowwise, this is done by reshaping into the local shape, allgathering,
    and stacking rows. For colwise, the only difference is we stack columns.
    This is done via vstack and column_stack respectively.
    """
    mp_size = dist.get_world_size(model_parallel_group)
    ref_shape = ref_state_dict[fqn].shape
    assert (
        ref_shape[0] == tp_sharded_shape[0] or ref_shape[1] == tp_sharded_shape[1]
    ), f"Expected sharded shape to match either row or col-wise, but does not: {ref_shape} {tp_sharded_shape} for fqn {fqn}"
    _verify_fqn_across_ranks(fqn, grp_gloo)
    if ref_shape[0] != tp_sharded_shape[0]:
        assert (
            ref_shape[0] == tp_sharded_shape[0] * mp_size
        ), f"Shape mismatch - ref {ref_shape} vs {tp_sharded_shape} mp_size {mp_size} for fqn {fqn}"
        # reshape the flat data_tensor into the rowwise shape
        data_tensor = data_tensor.reshape(tp_sharded_shape)
        # now, all_gather such tensors
        tensor_list = _all_gather_into_list(data_tensor, model_parallel_group)
        # stack rowwise to produce the final unsharded tensor
        data_tensor = torch.vstack(tensor_list).cpu()
        assert data_tensor.shape == ref_shape
        full_shape = data_tensor.shape
    elif (
        len(ref_shape) > 1
        and len(tp_sharded_shape) > 1
        and ref_shape[1] != tp_sharded_shape[1]
    ):
        assert ref_shape[1] == mp_size * tp_sharded_shape[1]
        # first, reshape the flat data_tensor into the colwise shape
        data_tensor = data_tensor.reshape(tp_sharded_shape)
        tensor_list = _all_gather_into_list(data_tensor, model_parallel_group)
        data_tensor = torch.column_stack(tensor_list).cpu()
        assert data_tensor.shape == ref_shape, f"{data_tensor.shape} vs {ref_shape}"
        full_shape = data_tensor.shape
    else:
        assert (
            ref_shape == tp_sharded_shape
        ), f"Shape mismatch: {ref_shape} vs {tp_sharded_shape} for fqn {fqn}"  # not tensor parallel sharded
        full_shape = tp_sharded_shape
        logging.warning(f"{fqn} {ref_shape} {full_shape} - not sharded")
    return data_tensor, full_shape


def get_consolidated_ckpt_path(
    ckpt_dir: Union[str, Path], mp_rank: int = 0, mp_size: int = 1
) -> Union[str, Path]:
    fname = "consolidated.00.pth" if mp_size == 1 else f"consolidated.0{mp_rank}.pth"
    return (
        ckpt_dir / fname
        if isinstance(ckpt_dir, Path)
        else os.path.join(ckpt_dir, fname)
    )


def _load_checkpoint(
    model, meta_model, model_parallel_size: int, ckpt_dir: str
) -> None:
    mp_group, _ = dist.new_subgroups(group_size=model_parallel_size)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        raise RuntimeError("Expected local_rank to be set, but it is not!")
    mp_rank = local_rank % model_parallel_size
    state_dict_pth = get_consolidated_ckpt_path(
        ckpt_dir=ckpt_dir, mp_rank=mp_rank, mp_size=model_parallel_size
    )
    state_dict = torch.load(state_dict_pth)
    dist_state_dict = build_distributed_state_dict_from_consolidated(
        meta_model,
        state_dict,
        model_parallel_world_size=model_parallel_size,
        use_dtensor=False,
    )
    if dist.get_world_size() == 1:
        # Use full_state_dict, so convert to torch.Tensor
        for k, v in dist_state_dict.items():
            if isinstance(v, ShardedTensor):
                dist_state_dict[k] = v.local_shards()[0].tensor
    log.debug("build distributed_state_dict")
    missing_keys, unexpected_keys = model.load_state_dict(dist_state_dict, strict=False)
    assert not missing_keys
    assert len(unexpected_keys) == 1 and "freqs" in unexpected_keys[0]


def build_distributed_state_dict_from_consolidated(
    model: nn.Module,
    consolidated_state_dict: Dict[str, Tensor],
    offload_to_cpu: bool = False,
    use_dtensor: bool = False,
    model_parallel_world_size: int = 8,
) -> Dict[str, Union[Tensor, DTensor, ShardedTensor]]:
    """
    Main API that takes a model (with no parallelism applied) and a fairscale checkpoint
    and builds a PT-D compliant distributed state dict. Note that this expects a consolidated
    checkpoint.

    Args:
        model (torch.nn.Module): module with no parallelism applied
            (i.e. result of `build_model` with parallel_impl=ParallelImpl.NONE)
        fs_state_dict (Dict[str, Any]): Fairscale consolidated
        offload_to_cpu (bool): Whether to offload the resulting state_dict to CPU (default: False)
        use_dtensor (bool): Whether to use PyTorch Distributed Tensor instead of ShardedTensor (default: False)
            (this will eventually default to True)
        model_parallel_world_size: Model parallel world size that was used to create the consolidated checkpoint.
            This can be obtained by checking the number of consolidated0x.pth files in the checkpoint directory.

    Example usage::
        ```
        ckpt_path = "path_to_llama"
        MODEL_PARALLEL_SIZE = 8
        state_dict = torch.load(ckpt_path)
        # Build a local LLaMA with no parallelism
        model = build_model(...)
        sharded_state_dict = build_distributed_state_dict_from_consolidated(
            model, state_dict, model_parallel_world_size=MODEL_PARALLEL_SIZE,
        )
        # Wrap model with PT-native APIs + load
        model = FSDP(model)
        FSDP.set_state_dict_type(StateDictType.SHARDED_STATE_DICT)
        model.load_state_dict(sharded_state_dict)
        ```

    Note: Please make sure to pass an unsharded model as the model arg! Otherwise, things will not
    work.

    This distributed state dict is a mapping of FQN: ShardedTensor/DTensor. It will be replaced with
    DTensor once DTensor 2D checkpoint format is fully rolled out.

    Note: This has only been tested for loading state_dict into PT-D FSDP sharded_state_dict for now.
    """
    torch._C._log_api_usage_once("build_distributed_state_dict")
    dist_state_dict = {}
    ref_state_dict = model.state_dict()
    grp_gloo = dist.new_group(backend="gloo")
    # TODO: this should be the FSDP device mesh
    mesh = (
        DeviceMesh(
            device_type="cuda",
            mesh=list(range(dist.get_world_size())),
        )
        if use_dtensor
        else None
    )
    input_dtypes = {v.dtype for v in consolidated_state_dict.values()}
    logging.warning(f"input_dtypes {input_dtypes}")
    model_parallel_group, _ = dist.new_subgroups(group_size=model_parallel_world_size)
    for fqn, tensor in consolidated_state_dict.items():
        # Hack for buffer
        if "rope.freqs" in fqn:
            dist_state_dict[fqn] = tensor.clone()
            continue
        if _is_tp_sharded(fqn):
            tensor, _ = _unshard_param(
                ref_state_dict,
                fqn,
                model_parallel_group,
                grp_gloo,
                tensor,
                tensor.shape,
            )
        if use_dtensor:
            assert mesh is not None
            tensor = _ext_chunk_dtensor(
                tensor=tensor.contiguous(),
                rank=dist.get_rank(),
                device_mesh=mesh,
            )
        else:
            tensor = _ext_chunk_tensor(
                tensor=tensor.contiguous(),
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                num_devices_per_node=torch.cuda.device_count(),  # TODO: this is not accurate if user set CUDA_VISIBLE_DEVICES
                pg=dist.distributed_c10d._get_default_group(),  # TODO: this should be the FSDP process group
            )
        dist_state_dict[fqn] = tensor

    dtypes = {v.dtype for v in dist_state_dict.values()}
    logging.warning(f"Made dist_state_dict with dtypes {dtypes}")
    return dist_state_dict


def main():
    from torch.distributed import init_process_group

    init_process_group(backend="nccl")
    mp_size = dist.get_world_size()
    model_size = "7b" if mp_size == 1 else "70b"
    ckpt_dir = f"/data/users/rvarm1/llama2-{model_size}/"
    # torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(dist.get_rank())
    model_parallel_size = min(8, dist.get_world_size())
    model_parallel_size = mp_size
    mp_group, _ = dist.new_subgroups(group_size=model_parallel_size)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        raise RuntimeError("Expected local_rank to be set, but it is not!")
    mp_rank = local_rank % model_parallel_size
    from dataclasses import dataclass

    #     def args_70b(
    #     max_seq_len: int = _DEFAULT_MAX_SEQ_LEN,
    #     max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
    # ) -> ModelArgs:
    #     """
    #     Returns args for LLaMA-70B.
    #     """
    #     return ModelArgs(
    #         dim=8192,
    #         hidden_layer_dim_multiple_of=4096,
    #         ffn_dim_multiplier=1.3,
    #         n_heads=64,
    #         n_kv_heads=8,
    #         n_layers=80,
    #         vocab_size=_DEFAULT_EMBEDDING_DIM,
    #         max_seq_len=max_seq_len,
    #         max_batch_size=max_batch_size,
    #     )

    from typing import Optional

    from tests.torchtune.models.llama2.scripts.compare_decoder import Transformer
    from tests.torchtune.models.llama2.scripts.compare_decoder_layer import (
        TransformerBlock,
    )

    @dataclass
    class LlamaArgs:
        vocab_size: int
        num_layers: int
        num_heads: int
        num_kv_heads: int
        embed_dim: 4096
        max_seq_len: int
        hidden_layer_dim_multiple_of: int = 256
        ffn_dim_multiplier: Optional[float] = None

    def args_70b():
        return LlamaArgs(
            vocab_size=32_000,
            num_layers=80,
            num_heads=64,
            num_kv_heads=8,
            embed_dim=8192,
            max_seq_len=2048,
            hidden_layer_dim_multiple_of=4096,
            ffn_dim_multiplier=1.3,
        )

    def args_7b():
        return LlamaArgs(
            vocab_size=32_000,
            num_layers=32,
            num_heads=32,
            num_kv_heads=32,
            embed_dim=4096,
            max_seq_len=4096,
            hidden_layer_dim_multiple_of=256,
            ffn_dim_multiplier=None,
        )

    from torchtune.models.llama2 import llama2

    def build_torchtune_model(model_size):
        assert model_size in ["7b", "70b"]
        args = args_7b() if model_size == "7b" else args_70b()
        return llama2(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            embed_dim=args.embed_dim,
            max_seq_len=args.max_seq_len,
            hidden_layer_dim_multiple_of=args.hidden_layer_dim_multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )

    def build_ref_model(model_size):
        assert model_size in ["7b", "70b"]
        args = args_7b() if model_size == "7b" else args_70b()
        return Transformer(
            vocab_size=args.vocab_size,
            dim=args.embed_dim,
            n_layers=args.num_layers,
            n_heads=args.num_heads,
            n_kv_heads=args.num_kv_heads,
            max_seq_len=args.max_seq_len,
            hidden_layer_dim_multiple_of=args.hidden_layer_dim_multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )

    torch.manual_seed(0)
    with torch.device("meta"):
        meta_model = build_ref_model(model_size)

    from tests.torchtune.models.llama2.scripts.compare_attention import (
        precompute_freqs_cis,
    )

    # meta
    # meta_model.freqs_cis = precompute_freqs_cis(
    #     8192 // 64, 2048 * 2
    # )

    # import pdb ; pdb.set_trace()
    state_dict_pth = get_consolidated_ckpt_path(
        ckpt_dir=ckpt_dir, mp_rank=mp_rank, mp_size=model_parallel_size
    )
    state_dict = torch.load(state_dict_pth)
    # dist_state_dict = {}
    # Trim
    # keys_to_del = []
    # for k in state_dict.keys():
    #     try:
    #         if int(k.split(".")[1]) >= 2:
    #             keys_to_del.append(k)
    #     except Exception as e:
    #         print(f"Not deleting key - {k}")
    # for k in keys_to_del:
    #     del state_dict[k]

    # print(f"RV: Deleted {len(keys_to_del)} keys", flush=True)
    fair_ckpt_dtype_set = set(v.dtype for v in state_dict.values())
    print(f"RV: fair_ckpt_dtype_set: {fair_ckpt_dtype_set}")
    if True or dist.get_world_size() > 1:
        dist_state_dict = build_distributed_state_dict_from_consolidated(
            meta_model,
            state_dict,
            model_parallel_world_size=model_parallel_size,
            use_dtensor=False,
        )
        if dist.get_world_size() == 1:
            # Use full_state_dict, so convert to torch.Tensor
            for k, v in dist_state_dict.items():
                if isinstance(v, ShardedTensor):
                    dist_state_dict[k] = v.local_shards()[0].tensor
                    dist_state_dict[k] = dist_state_dict[k].cpu()
    else:
        dist_state_dict = state_dict

    print(f"RV: about to map weights")
    torchtune_sd = _convert_llama_from_fair("blah", dist_state_dict)
    print(f"RV: DONE mapping weights, now converting state_dict")
    # from torch.distributed._state_dict_utils import _gather_state_dict as gsd
    # # gsd(torchtune_sd) ; gsd(dist_state_dict)
    # dist.barrier()
    # torchtune_sd = gsd(torchtune_sd)
    # dist.barrier()
    # dist_state_dict = gsd(dist_state_dict)
    # dist.barrier()
    # print(f"RV: DONE gathering full SDs ")
    torch.manual_seed(0)
    with torch.device("meta"):
        torchtune_model = build_torchtune_model(model_size)

    # torchtune_model.to_empty(device=torch.cuda.current_device())
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from torchtune.modules.transformer import TransformerDecoderLayer

    # for m in torchtune_model.modules():
    #     try:
    #         m._rope_init(device=torch.cuda.current_device())
    #     except AttributeError:
    #         pass
    torch.manual_seed(0)
    from functools import partial

    from torchtune.modules.peft.peft_utils import lora_fsdp_init

    aasdfjkll = partial(lora_fsdp_init, device=torch.cuda.current_device())
    from torch.distributed.fsdp import CPUOffload

    torchtune_model = FSDP(
        torchtune_model,
        auto_wrap_policy=ModuleWrapPolicy({TransformerDecoderLayer}),
        device_id=torch.cuda.current_device(),
        param_init_fn=aasdfjkll,
        # use_orig_params=True,
        # cpu_offload=CPUOffload(offload_params=True),
    )
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
    )
    apply_activation_checkpointing(torchtune_model, auto_wrap_policy=ModuleWrapPolicy({TransformerDecoderLayer}))

    # torchtune_model.eval()
    from torch.distributed.fsdp.api import StateDictType

    sd_type = (
        StateDictType.SHARDED_STATE_DICT
        if dist.get_world_size() > 1
        else StateDictType.FULL_STATE_DICT
    )
    FSDP.set_state_dict_type(torchtune_model, sd_type)
    missing, unexpected = torchtune_model.load_state_dict(torchtune_sd, strict=False)
    for k, v in torchtune_model.state_dict().items():
        if v.dtype != torch.bfloat16:
            print(f"Expected {k} to be bf16, got {v.dtype}")
    print(f"RV: missing {missing} unexpected {unexpected}")
    inp = torch.randint(0, 32_000, (1, 256)).to(torch.cuda.current_device())
    # optim = torch.optim.AdamW(torchtune_model.parameters(), lr=0.03, foreach=False)
    # import bitsandbytes as bnb
    # optim = bnb.optim.PagedAdamW(torchtune_model.parameters(), lr=0.03)
    # # optim = torch.optim.SGD(torchtune_model.parameters(), lr=0.03)
    # out = torchtune_model(inp).sum().backward()
    # optim.step()
    # print(f"RV: done w/backward + step")
    # exit(0)
    # with FSDP.summon_full_params(torchtune_model):
    #     from copy import deepcopy
    #     torchtune_model = deepcopy(torchtune_model.module)
    with torch.no_grad():
        out = torchtune_model(inp)

    del torchtune_model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    with torch.device("meta"):
        meta_model = build_ref_model(model_size)

    from tests.torchtune.models.llama2.scripts.compare_attention import (
        precompute_freqs_cis,
    )

    torch.set_default_device(torch.cuda.current_device())
    args = args_7b() if model_size == "7b" else args_70b()
    meta_model.freqs_cis = precompute_freqs_cis(
        args.embed_dim // args.num_heads, args.max_seq_len * 2
    )
    # assert meta_model.freqs_cis.device == torch.cuda.current_device(), f"Expected {torch.cuda.current_device()} got {meta_model.freqs_cis.device}"
    ref_model = FSDP(
        meta_model,
        auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}),
        device_id=torch.cuda.current_device(),
        param_init_fn=lambda m: m.to_empty(
            device=torch.cuda.current_device(), recurse=False
        ),
        use_orig_params=True,
    )
    # ref_model = meta_model
    # ref_model = meta_model
    # ref_model.eval()
    # FSDP.set_state_dict_type(ref_model, sd_type)
    FSDP.set_state_dict_type(ref_model, sd_type)
    missing, unexpected = ref_model.load_state_dict(dist_state_dict, strict=False)
    # with FSDP.summon_full_params(ref_model):
    #     from copy import deepcopy
    #     ref_model = deepcopy(ref_model.module)
    for k, v in ref_model.state_dict().items():
        if v.dtype != torch.bfloat16:
            print(f"Expected {k} to be bf16, got {v.dtype}")
    print(f"RV: [REF] missing {missing} unexpected {unexpected}")
    inp = inp.clone()
    with torch.no_grad():
        out2 = ref_model(inp)

    # assert torch.allclose(out, out_ref, atol=1e-5, rtol=1e-3)
    if dist.get_rank() == 0:
        fair_output = out2.sum()
        native_output = out.sum()
        torch.testing.assert_close(
            native_output,
            fair_output,
            rtol=1e-5,
            atol=1e-8,
            msg=f"[In validation] Outputs differ. FAIR output: {fair_output}. Native output: {native_output}",
        )
        print(f"Got parity {native_output} {fair_output}")
        # if not torch.allclose(out, out2, atol=1e-4):
        #     print(f"RV: {out} vs {out2}")

    # print(f"RV: {out} vs {out2}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    sys.exit(main())
    # parser.add_argument("--model_parallel_size", type=int, default=8)
    model_parallel_size = dist.get_world_size()
    sys.exit(main(args.model_parallel_size))
