import os
import pdb
from functools import partial

import ray
import torch
import torch.nn as nn
import torchtune
import torchtune.training as training
from torchtune import utils
from torchtune.models import qwen2_5


def setup_model(device, dtype, compile_model=False, cpu_offload=False):
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
    )

    with training.set_default_dtype(dtype), device:
        for m in model.modules():
            # RoPE is not covered in state dict
            if hasattr(m, "rope_init"):
                m.rope_init()

    model_sd = torchtune.training.FullModelHFCheckpointer(
        checkpoint_dir="/tmp/Qwen2.5-3B-Instruct",
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


@ray.remote(num_cpus=16, num_gpus=2)
class Generator:
    def __init__(self):
        self.device = torch.device("cuda")
        self.model = nn.Linear(5, 3, device="cuda")
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


# Define the worker class
@ray.remote(num_cpus=16, num_gpus=2)
class Trainer:
    def __init__(self):
        # Need to setup FSDP and multple workers?
        self.device = torch.device("cuda")  # utils.get_device(device="cuda")
        self.dtype = training.get_dtype("bf16", device=self.device)

        model = setup_model(self.device, self.dtype)
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
ray.init(num_cpus=32, num_gpus=4)
# Create two workers
trainer = Trainer.remote()  # [TODO] Launch FSDP with the worker
generator = Generator.remote()

# # Run the update method on both workers asynchronously
# ray.get([worker1.update.remote(), worker2.update.remote()])

# # # Get the state dictionary from worker 1
# state_dict = ray.get(worker1.get_state_dict.remote())

# # # Set the state dictionary on worker 2
# ray.get(worker2.set_state_dict.remote(state_dict))
ray.shutdown()
