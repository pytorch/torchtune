import os
import socket
from functools import partial

import ray
import torch
import torch.nn as nn
import torchtune
import torchtune.training as training
from torchtune import utils
from torchtune.models import qwen2_5

from vllm import LLM, SamplingParams
from vllm.worker.worker import Worker

from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


def stateless_init_process_group(master_address, master_port, rank, world_size,
                                 device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes) 
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(host=master_address,
                                      port=master_port,
                                      rank=rank,
                                      world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class MyWorker(Worker):
    """
    The `MyWorker` class inherits from `Worker` to provide custom functions.
    For simplicity, we define the `MyWorker` class in this self-contained 
    script. Normally, we should define the `MyWorker` class in a separate 
    file and pass the qualified name of the class to the `worker_cls` 
    parameter.
    """

    def init_weight_update_group(self, master_address, master_port,
                                 rank_offset, world_size):
        from vllm.distributed.parallel_state import get_world_group
        rank = get_world_group().rank + rank_offset
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(weight,
                                          src=0,
                                          stream=torch.cuda.current_stream())

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(
                p, torch.zeros_like(p))
        return weights_updated


class MyLLM(LLM):

    def __init__(self, *args, **kwargs):
        # a hack to make the script work.
        # stop ray from manipulating CUDA_VISIBLE_DEVICES
        # at the top-level
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)


def setup_model(device, dtype, compile_model=False, cpu_offload=False):
    import torch.distributed
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    world_size = torch.distributed.get_world_size()
    from torch.distributed.device_mesh import init_device_mesh
    device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

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
        dp_mesh = device_mesh['fsdp']
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



# Define the worker class
@ray.remote(num_cpus=16, num_gpus=1)
class TrainWorker:
    def __init__(self, environment_variables):
        import os
        for var in environment_variables:
            os.environ[var] = environment_variables[var]
        # Need to setup FSDP and multple workers?
        self.device = torch.device("cuda")  # utils.get_device(device="cuda")
        self.dtype = training.get_dtype("bf16", device=self.device)

        self.model = setup_model(self.device, self.dtype)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    async def update(self):
        # Create dummy data
        inputs = torch.randn(10, 5, device=self.device)
        labels = torch.randn(10, 3, device=self.device)

        # Run a single optimizer update
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = torch.mean((outputs - labels) ** 2)
        loss.backward()
        self.optimizer.step()

    async def get_state_dict(self):
        return self.model.state_dict()

    async def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


# Launch the Ray cluster with "ray start --head --num-gpus 2"
# To kill server run "ray stop"
ray.init(num_cpus=32, num_gpus=6)

fsdp_world_size = 4

def _get_node_ip():
    def get_node_ip_by_sdk():
        return ray._private.services.get_node_ip_address()

    host_ipv4 = os.getenv("MY_HOST_IP", None)
    host_ipv6 = os.getenv("MY_HOST_IPV6", None)
    host_ip_by_env = host_ipv4 or host_ipv6
    host_ip_by_sdk = get_node_ip_by_sdk()

    host_ip = host_ip_by_env or host_ip_by_sdk
    return host_ip

def _get_free_port():
    with socket.socket() as sock:
        sock.bind(('', 0))
        return sock.getsockname()[1]

def get_available_master_addr_port():
    return _get_node_ip(), str(_get_free_port())

master_address, master_port = get_available_master_addr_port()
fsdp_workers = []
for i in range(fsdp_world_size):
    env_vars = {
        'WORLD_SIZE': str(fsdp_world_size),
        'RANK': str(i),
        'WG_BACKEND': 'ray',
        'MASTER_ADDR': master_address,
        'MASTER_PORT': master_port,
    }

    # Launch the worker
    worker = TrainWorker.remote(env_vars)
    fsdp_workers.append(worker)

pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

llm = ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=scheduling_inference,
    )(LLM).remote(
        model="Qwen/Qwen2.5-3B",
        enforce_eager=True,
        worker_cls=MyWorker,
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
    )

# Generate texts from the prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)

outputs = ray.get(llm.generate.remote(prompts, sampling_params))

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, "
        f"Generated text: {generated_text!r}")

# # Run the update method on both workers asynchronously
# ray.get([worker1.update.remote(), worker2.update.remote()])

# # # Get the state dictionary from worker 1
# state_dict = ray.get(worker1.get_state_dict.remote())

# # # Set the state dictionary on worker 2
# ray.get(worker2.set_state_dict.remote(state_dict))
ray.shutdown()
