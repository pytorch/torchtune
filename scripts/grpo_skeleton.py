"""
Example use of an ever-running, fully async, distributed collector
==================================================================

This example demonstrates how to set up and use a distributed collector
with Ray in a fully asynchronous manner. The collector continuously gathers
data from a gym environment and stores it in a replay buffer, allowing for
concurrent processing and data collection.

Key Components:
1. **Environment Factory**: A simple function that creates instances of the
   `GymEnv` environment. In this example, we use the "Pendulum-v1" environment.
2. **Policy Definition**: A `TensorDictModule` that defines the policy network.
   Here, a simple linear layer is used to map observations to actions.
3. **Replay Buffer**: A `RayReplayBuffer` that stores collected data for later
   use, such as training a reinforcement learning model.
4. **Distributed Collector**: A `RayCollector` that manages the distributed
   collection of data. It is configured with remote resources and interacts
   with the environment and policy to gather data.
5. **Asynchronous Execution**: The collector runs in the background, allowing
   the main program to perform other tasks concurrently. The example includes
   a loop that waits for data to be available in the buffer and samples it.
6. **Graceful Shutdown**: The collector is shut down asynchronously, ensuring
   that all resources are properly released.

This setup is useful for scenarios where you need to collect data from
multiple environments in parallel, leveraging Ray's distributed computing
capabilities to scale efficiently.

Setup:
- `$ git clone https://github.com/pytorch/rl`
- `$ git clone https://github.com/pytorch/tensordict`
- `$ cd tensordict && python setup.py develop`
- `$ cd ../rl && git checkout grpo-ray && python setup.py develop`

"""
import asyncio
from functools import partial

import tensordict
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchrl.data.replay_buffers.ray_buffer import RayReplayBuffer
from torchrl.modules.llm.transformers_policy import from_hf_transformers
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from torchrl.collectors.distributed.ray import DEFAULT_RAY_INIT_CONFIG, RayCollector
from torchrl.envs import LLMEnv


async def main():
    # 1. Create environment factory

    # Setup
    # TODO: This is some super quickly loaded model, just to get the transformers signature right
    #  We will want to substitute that with a vLLM instance.
    #  We're working on a wrapper here: https://github.com/pytorch/rl/pull/2830
    #  Soon to be merged in the grpo-ray branch
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel(GPT2Config())

    # TODO: super basic dataloader on IMDB - the only thing that is important here is the format: we output
    #  a dict with tokens and attention_mask
    class IMDBDataset(Dataset):
        def __init__(self, dataset, tokenizer):
            self.dataset = dataset
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            text = self.dataset[idx]['text']
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            return {
                'tokens': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
            }

    def get_dataloader():
        dataset = load_dataset('imdb', split='train')
        dataset = IMDBDataset(dataset, tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        return dataloader

    dl = get_dataloader()
    print('dataloader', dl)

    # TODO: We build the env. If the dataloader has the right format we should be good.
    #  We need to make sure that we can share the dataloader across workers. I guess it's
    #  something tune folks know how to do, personally I never dealt with that.
    #  The basic idea is that LLMEnv will call next(dataloader) whenever it resets.
    #  The number of repeats num_repeats tells the env how many times we want to see the same
    #  occurence of a single prompt.
    # TODO: We will need some mechanism to aggregate trajectories with the same prompt for adv computation
    #  if the results come from different workers.
    env_maker = partial(
        LLMEnv.from_dataloader,
        dl,
        device="cpu", # TODO: we should have proper default values for these
        no_stack=True,
        batch_size=1, # zero-rewards of shape identical to tokens
        assign_reward=True,
        has_attention=True, # TODO: this can be inferred automatically
        num_repeats=4,
        )

    # TODO: make this one work too:
    #  policy = from_vllm_transformers(model)
    # For large models we may want to refactor the logic
    policy = from_hf_transformers(model)
    # Print the size of the model
    print('model size', tensordict.from_module(policy).bytes() / (1024 ** 2), "Mb")

    # We run a check env + policy, can be removed
    print("example data", env_maker().rollout(1, policy))
    print('env', env_maker())

    # Create the buffer
    # TODO: we want to customize this buffer, having hooks for reward and ref model computation.
    #  This version just writes plain data in the buffer and samples the unaltered data.
    remote_config = {
        "num_cpus": 4,  # Assuming a model with an 8-core CPU
        "num_gpus": 0,  # Most MacBook Pros do not have discrete GPUs suitable for deep learning
        "memory": 4 * 1024 ** 3,  # Assuming 8 GB of RAM
        "object_store_memory": 1024 ** 3,  # Allocate 4 GB for object store memory
    }
    ray_init_kwargs = DEFAULT_RAY_INIT_CONFIG
    ray_init_kwargs.update(
        {
            "num_cpus": 10, "num_gpus": 0, "object_store_memory": 2 * 1024 ** 3,
            # "memory": 16 * 1024 ** 3,
        }
    )
    buffer = RayReplayBuffer(ray_init_kwargs=ray_init_kwargs)

    # 2. Define distributed collector
    # TODO: what's missing here is the async env logic. It's super cool but can be fixed later. Buffer, vllm and others
    #  need to be fixed before.
    distributed_collector = RayCollector(
        [env_maker], policy, total_frames=4, frames_per_batch=1, remote_configs=remote_config, replay_buffer=buffer, )

    # We start the data collection
    print("start")
    distributed_collector.start()

    # TODO: This should be abstracted under a trainer.train() call that starts the trainer node.
    # TODO: decide where and how the weight sync happens. To me it makes more sense for the collector node (inference)
    #  to ask for weights at regular intervals, than it would be for the trainer to send these. Or do we have some
    #  shared access to a weight pool where the weights are written?
    #  If the logic is to be implemented on the node side we need to make this happen in the collector
    while True:
        while not len(buffer):
            print("waiting")
            await asyncio.sleep(1)  # Use asyncio.sleep instead of time.sleep
        print("sample", buffer.sample(32))
        # break at some point
        break

    await distributed_collector.async_shutdown()


if __name__ == "__main__":
    asyncio.run(main())
