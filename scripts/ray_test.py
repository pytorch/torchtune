import os

import ray
import torch
import torch.nn as nn


# Define the worker class
@ray.remote(num_cpus=8, num_gpus=1)
class Worker:
    def __init__(self):
        self.model = nn.Linear(5, 3)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        # device
        gpu_ids = ray.get_gpu_ids()
        self.device = torch.device(f"cuda:{gpu_ids[0]}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        breakpoint()
        # self.model.to(self.device)

    # async
    def update(self):
        # Create dummy data
        inputs = torch.randn(10, 5).to(self.device)
        labels = torch.randn(10, 3).to(self.device)
        # Run a single optimizer update
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = torch.mean((outputs - labels) ** 2)
        loss.backward()
        self.optimizer.step()

    # async
    def get_state_dict(self):
        return self.model.state_dict()

    # async
    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


# Launch the Ray cluster with "ray start --head --num-gpus 2"
# To kill server run "ray stop"
ray.init(num_cpus=16, num_gpus=2)
# Create two workers
worker1 = Worker.remote()
worker2 = Worker.remote()

# Run the update method on both workers asynchronously
ray.get([worker1.update.remote(), worker2.update.remote()])

# # Get the state dictionary from worker 1
# state_dict = ray.get(worker1.get_state_dict.remote())

# # Set the state dictionary on worker 2
# ray.get(worker2.set_state_dict.remote(state_dict))
ray.shutdown()
