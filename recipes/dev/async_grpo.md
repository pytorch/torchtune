# Distributed Async GRPO

The Group Relative Policy Optimization (GRPO) Recipe is an RL technique for post-training LLMs through rewards. torchtune supports grpo_full_finetune_distributed and async_grpo_full_finetune_distributed. grpo_full_finetune_distributed is based on the original implementation, for a deeper understanding on how GRPO works you can view the [paper](https://arxiv.org/pdf/2402.03300). But the primary difference between the two recipes is how training and generation is sequenced. In the original paper, training and generation happen in turns.

```
|-- generate --||shard||-- train --||shard||-- generate --||shard||-- train --||shard||-- generate --||shard||-- train --|
```

This allows you to use the same hardware for both training and generation but means that you're only using your resources for training a fraction of the time as you have to spend a lot of time generating your datasets through decoding. In the async approach, we overlap generation and training in order to maximize training speed. This requires you to split your hardware between training and generation GPUs but allows you to choose and allocation that balances training time vs decoding time. In the case where you keep the trainer on-policy, you already can see a big advantage in your ability to efficiently use your resources.

```
|-- generate --|      |-- generate --|      |-- generate --|
    |-- train --||sync|   |-- train --||sync|   |-- train --|
```

But you can also take this further by allowing your generation to go off-policy by a controlled amount. This allows your generator to keep running until the trainer updates it, which also ensures there's extra data queued up for the trainer to immediately continue training once the generator is updated. The goal of this recipe is to give flexibility to users to find the optimal tradeoff between GPU utilization and off-policy tolerance.

```
|----- generate -----||--- generate ----||- generate -|
    |-- train --||sync||-- train --||sync||-- train --|
```


> **WARNING** This recipe is still in a very early stage and under active development. There will be many breaking changes over the comming weeks and you should view this as a preview only. The initial version of this recipe is optimized around Qwen 2.5 3B to be able to benchmark and test all of the components. Over time this will be integrated with all of the expected torchtune features.


## Architecture

At a high level, the recipe consists of multiple "workers" all coordinated using Ray. Here are the primary ones to understand.

- Trainer - this is essentially the same as our exisitng "full_finetune_distributed" recipes but instead of iterating over a dataloader it iterates over a `torchrl.data.ReplayBuffer` which is populated by the generator.
- Generator - This is a special instance a `torchrl.collectors.SyncDataCollector`. The data collector manages running a `vLLM` instance paired with a batch of environments that the vLLM workers act on. The `environments` can be interactive environments or as simple as a dataset that returns prompts. After a run is complete, a GRPOTrajectory object is returned and pushed to the ReplayBuffer for the trainer to pickup. There is some post processing that happens on the trajectory with the current split in responsiblities below:
   a. query_responses: vllm worker computes
   b. logprobs vllm worker computes
   c. ref_logprobs: computed by reference model after vllm workers
   d. rewards: trainer worker computes
   e. sucecesses: trainer worker computes
   f. advantages: trainer worker computes
   g. masks: trainer worker computes
   h. position_ids: trainer worker computes
- VLLMParameterServer - this maintains a copy of the model parameters and is used to manage syncing between the trainer and generator. The trainer pushes to the server every n steps, while the generator checks for updates every m steps and will pull updates to vLLM wokers. ** Currently this only supports single GPU so we're limited to models that can fit on a single GPU **

All of the above are run as remote Ray workers, with Ray being used for managing resource allocation.

## Using the Recipe

We recommend installing this in a dedicated conda environment.
The following works for us:

```bash
conda create --name tunerl python=3.10
conda activate tunerl
git clone https://github.com/joecummings/r1-zero.git
cd r1-zero
pip install torch torchvision torchao
pip install -e .[async_rl]

```

With these installed you can run the recipe with

```bash
tune run dev/async_grpo_full_finetune_distributed --config recipes/configs/dev/qwen3B_async_grpo.yaml
```

Apart from the standard config options, GRPO introduces
- grpo_samples
- forward_batch_size
- max_generated_tokens
- top_k
- temperature
- replay_buffer_size
- ppo_epochs
- num_steps
- steps_before_sync
- num_ref_workers
- num_fsdp_workers
- vllm
   num_workers
   tp_size
   batch_size
   steps_before_sync
   queue_maxsize
