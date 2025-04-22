import os
import socket
from functools import partial

import ray
import torch
import torch.nn as nn
import torchtune
import torchtune.training as training

from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torchtune import utils
from torchtune.models import qwen2_5
from torchtune.models.qwen2._convert_weights import qwen2_tune_to_hf

from vllm import LLM, SamplingParams
from vllm.worker.worker import Worker


def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: torch.device,
):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


# [todo] src == 1? Not 0?
# [todo] rank is local to ray worker or global?
# [todo] def check_weights_changed(self)?
class vLLMRayWorker(Worker):
    """vLLM worker for Ray."""

    def init_weight_update_group(self, master_address, master_port, rank, world_size):
        from vllm.distributed.parallel_state import get_world_group

        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(
            weight, src=1, stream=torch.cuda.current_stream()
        )
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight


# Define the worker class
@ray.remote(num_cpus=16, num_gpus=1)
class TrainWorker:
    def __init__(self, environment_variables):
        # Need to setup FSDP and multple workers?
        self.device = utils.get_device(device="cuda")
        self.dtype = training.get_dtype("bf16", device=self.device)
        import os

        for var in environment_variables:
            os.environ[var] = environment_variables[var]

        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        self.rank = os.environ["RANK"]
        self.world_size = int(os.environ["WORLD_SIZE"])

        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )
        print(self.device_mesh)
        self.model = self.setup_model(self.device, self.dtype)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def zero_weights(self):
        for name, p in self.model.named_parameters():
            # .data to prevent leaf Variable in-place modification RuntimeError
            p.data.zero_()
        return True

    def get_state_dict(self):
        return self.model.state_dict()

    def init_model_update_group(self, master_address, master_port, rank, world_size):
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            # FIXME: hardcoding, not sure if this is right
            torch.device(f"cuda:0"),
        )

    def get_metadata_for_broadcast(self):
        vllm_format_sd = self.new_sd
        new_sd = {}
        for k, v in vllm_format_sd.items():
            new_sd[k] = (v.shape, v.dtype)
        return new_sd

    def all_gather(self):
        new_sd = {}
        for i, (k, v) in enumerate(self.model.state_dict().items()):
            new_sd[k] = v.full_tensor()
            if i == 0:
                print(
                    f"DTensor.local shape {v._local_tensor.shape}, DTensor.full_tensor shape {new_sd[k].shape}"
                )
        new_sd = qwen2_tune_to_hf(new_sd, num_heads=16, num_kv_heads=2, dim=2048)
        # FIXME: is this sus
        self.new_sd = new_sd

    def broadcast_key_to_vllm(self, key):
        self.model_update_group.broadcast(
            self.new_sd[key], src=1, stream=torch.cuda.current_stream()
        )

    def get_rank(self):
        gpu_ids = ray.get_gpu_ids()
        print(gpu_ids)
        return gpu_ids[0]

    def setup_model(self, device, dtype, compile_model=False, cpu_offload=False):
        with training.set_default_dtype(dtype), torch.device("meta"):
            model = qwen2_5.qwen2_5_3b()

        if compile_model:
            training.compile_model(model, verbose=False)

        # For FSDP sharding
        fsdp_shard_conditions = [partial(training.get_shard_conditions)]

        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=cpu_offload,
            reshard_after_forward=True,
            dp_mesh=self.device_mesh["fsdp"],
        )

        with training.set_default_dtype(dtype), device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        model_sd = torchtune.training.FullModelHFCheckpointer(
            checkpoint_dir="/tmp/Qwen2.5-3B",
            checkpoint_files=[
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
            recipe_checkpoint=None,
            output_dir="/tmp/torchtune/qwen2_5_3B/ray_vllm_test",
            model_type="QWEN2",
        ).load_checkpoint()[training.MODEL_KEY]

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            model_sd,
            device,
            strict=True,
            cpu_offload=cpu_offload,
        )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)
        training.disable_dropout(model)
        return model


# Launch the Ray cluster with "ray start --head --num-gpus 2"
# To kill server run "ray stop"
ray.init(num_cpus=192, num_gpus=6)


# ====== init vllm ==========

# vllm_tp_size = 1

# pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * vllm_tp_size)
# ray.get(pg_inference.ready())
# scheduling_inference = PlacementGroupSchedulingStrategy(
#     placement_group=pg_inference,
#     placement_group_capture_child_tasks=True,
#     placement_group_bundle_index=0,
# )

# llm = ray.remote(
#     num_cpus=0,
#     num_gpus=0,
#     scheduling_strategy=scheduling_inference,
# )(LLM).remote(
#     model="Qwen/Qwen2.5-3B",
#     enforce_eager=True,
#     worker_cls=MyWorker,
#     tensor_parallel_size=vllm_tp_size,
#     distributed_executor_backend="ray",
# )

# Generate texts from the prompts.
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]

# sampling_params = SamplingParams(temperature=0)

# outputs = ray.get(llm.generate.remote(prompts, sampling_params))

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, " f"Generated text: {generated_text!r}")

# === init fsdp ====

# fsdp_world_size = 4


# def _get_node_ip():
#     def get_node_ip_by_sdk():
#         return ray._private.services.get_node_ip_address()

#     host_ipv4 = os.getenv("MY_HOST_IP", None)
#     host_ipv6 = os.getenv("MY_HOST_IPV6", None)
#     host_ip_by_env = host_ipv4 or host_ipv6
#     host_ip_by_sdk = get_node_ip_by_sdk()

#     host_ip = host_ip_by_env or host_ip_by_sdk
#     return host_ip


# def _get_free_port():
#     with socket.socket() as sock:
#         sock.bind(("", 0))
#         return sock.getsockname()[1]


# def get_available_master_addr_port():
#     return _get_node_ip(), str(_get_free_port())


master_address, master_port = get_available_master_addr_port()
fsdp_workers = []
for i in range(fsdp_world_size):
    env_vars = {
        "WORLD_SIZE": str(fsdp_world_size),
        "RANK": str(i),
        "WG_BACKEND": "ray",
        "MASTER_ADDR": master_address,
        "MASTER_PORT": master_port,
    }
    # print(env_vars)

    # Launch the worker
    worker = TrainWorker.remote(env_vars)
    fsdp_workers.append(worker)

fsdp_nodes = []
for worker in fsdp_workers:
    fsdp_nodes.append(ray.get(worker.get_rank.remote()))
chosen_fsdp_master_rank = fsdp_workers[fsdp_nodes[0]]

# rank = None
# for i in range(5):
#     if i in fsdp_nodes:
#         continue
#     else:
#         rank = i

# print(f"{rank=}")


# ensure there is not process group initialized in the main process
assert not torch.distributed.is_initialized()

# === init stateless process group for comms between vllm and fsdp workers ===
from vllm.utils import get_ip, get_open_port

weight_update_address = get_ip()
weight_update_port = get_open_port()
print(f"{weight_update_port=}")

handle = llm.collective_rpc.remote(
    "init_weight_update_group",
    args=(weight_update_address, weight_update_port, 0, vllm_tp_size + 1),
)

handle2 = chosen_fsdp_master_rank.init_model_update_group.remote(
    weight_update_address, weight_update_port, vllm_tp_size, vllm_tp_size + 1
)

# only need to .get one of the two handles since this will block until all
# participating ranks init
ray.get(handle)
print("initiated vllm weight update groups")


# === simulate training, modify the weights of the model. ===
for i, train_shard in enumerate(fsdp_workers):
    ray.get(train_shard.zero_weights.remote())


# === sync weight from the training process to the inference engine ===

# all gather on all ranks
handles = []
for worker in fsdp_workers:
    handle = worker.all_gather.remote()
    handles.append(handle)

[ray.get(handle) for handle in handles]
print("done all gather")

# get metadata for broadcast
metadata = ray.get(chosen_fsdp_master_rank.get_metadata_for_broadcast.remote())
print("got metadata")

# do broadcast key by key
for name, (shape, dtype) in metadata.items():
    handle = llm.collective_rpc.remote("update_weight", args=(name, dtype, shape))
    chosen_fsdp_master_rank.broadcast_key_to_vllm.remote(name)
    ray.get(handle)
    print(f"updated {name=}")

# check if the weights are updated to 0.
assert all(ray.get(llm.collective_rpc.remote("check_weights_changed")))

# use the updated model to generate texts, they will be nonsense
outputs = ray.get(llm.generate.remote(prompts, sampling_params))

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, " f"Generated text: {generated_text!r}")

ray.shutdown()


# [todo] missing train, setup, uses vllm example code instead of above
class RayGRPORecipe:
    def __init__(self, cfg):
        self.ray_resources = cfg.ray_resources
        self.vllm_tensor_parallel_size = cfg.vllm_tensor_parallel_size
        self.fsdp_world_size = cfg.fsdp_world_size

    def start_ray(self):
        ray.init(
            num_cpus=self.ray_resources.num_cpus, num_gpus=self.ray_resources.num_gpus
        )

    def setup(self):
        # ---- Create models ---- #
        self.vllm_model = _create_vllm_rollout(
            tensor_parallel_size=self.tensor_parallel_size
        )
        addr = get_ip()
        train_workers_port = get_open_port()
        self.pytorch_train_workers = _create_train_workers(
            world_size=self.fsdp_world_size, addr=addr, port=train_workers_port
        )
        # ---- Create PG to sync weights to rollout model ---- #
        # Ensure there is not process group initialized in the main process
        assert not torch.distributed.is_initialized()
        weight_update_port = get_open_port()
        self.vllm_model.collective_rpc.remote(
            "init_weight_update_group",
            args=(addr, weight_update_port, 0, self.vllm_tensor_parallel_size + 1),
        )
        self.pytorch_train_workers[0].init_model_update_group.remote(
            addr,
            weight_update_port,
            self.vllm_tensor_parallel_size,
            self.fsdp_world_size,
        )
        ray.get(
            handle
        )  # Wait for the weight update group to be initialized [todo] handle not defined in scope
        # ---- Create dataloader ---- #
        collate_fn = partial(
            _get_component_from_path(cfg.collate_fn),
            padding_idx=self._tokenizer.pad_id,
        )
        self.dataloader = _create_dataloader(
            collate_fn,
            batch_size=cfg.batch_size,
        )
        # ---- Create optimizer ---- #

    def _create_vllm_rollout(tensor_parallel_size: int):
        # Create placement group (still kinda need to figure out what this does)
        pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * tensor_parallel_size)
        ray.get(pg_inference.ready())
        scheduling_inference = PlacementGroupSchedulingStrategy(
            placement_group=pg_inference,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        # Initialize the remote model
        llm = ray.remote(
            num_cpus=0,
            num_gpus=0,
            scheduling_strategy=scheduling_inference,
        )(LLM).remote(
            model="Qwen/Qwen2.5-3B",
            enforce_eager=True,
            worker_cls=vLLMRayWorker,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="ray",
        )
        return llm

    def _create_train_workers(
        world_size: int, addr: str, port: int
    ) -> List[PyTorchTrainWorker]:
        workers = []
        for i in range(world_size):
            env_vars = {
                "WORLD_SIZE": str(world_size),
                "RANK": str(i),
                "WG_BACKEND": "ray",
                "MASTER_ADDR": addr,
                "MASTER_PORT": str(port),
            }
            worker = PyTorchTrainWorker.remote(env_vars)
            workers.append(worker)
        return workers

    def _create_dataloader(collate_fn, batch_size, ds, shuffle, rank, world_size):
        sampler = StatefulDistributedSampler(
            ds,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            seed=self.seed,
        )
        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=(),
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
        )
        return dataloader

    def _sync_weights():
        handles = []
        for worker in fsdp_workers:
            handle = worker.all_gather.remote()
            handles.append(handle)

        [ray.get(handle) for handle in handles]
        print("done all gather")

        # get metadata for broadcast
        metadata = ray.get(chosen_fsdp_master_rank.get_metadata_for_broadcast.remote())

        # do broadcast key by key
        for name, (shape, dtype) in metadata.items():
            handle = llm.collective_rpc.remote(
                "update_weight", args=(name, dtype, shape)
            )
            chosen_fsdp_master_rank.broadcast_key_to_vllm.remote(name)
            ray.get(handle)

    def stop_ray(self):
        ray.shutdown()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    recipe = RayGRPORecipe(cfg=cfg)
    config.log_config(recipe_name="RayGRPORecipe", cfg=cfg)
    recipe.start_ray()
    recipe.setup()
    recipe.train()
    recipe.cleanup()
    recipe.stop_ray()


if __name__ == "__main__":
    sys.exit(recipe_main())
