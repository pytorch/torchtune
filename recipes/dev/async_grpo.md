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
git clone https://github.com/pytorch/torchtune.git
cd torchtune
pip install torch torchvision torchao
pip install -e .[async_rl]

```
Before running this, you need to a) download the model file and b) be logged into Weights and Biases to track the experiment. So let's make sure of that:

```
conda activate tunerl
tune download Qwen/Qwen2.5-3B --output-dir /tmp/Qwen2.5-3B --ignore-patterns "original/consolidated.00.pth"
wandb login
```

Now everything should be taken care of! From your conda env, now you can run:

```bash
tune run dev/async_grpo_full_finetune_distributed --config dev/qwen3B_async_grpo
```

We can run the above successfully on a server using 8x H100 nodes. We left GPU memory to spare so we think it should work on cards with less VRAM, but if not we recommend reducing batch sizes to make this fit.

Note that we currently haven't focused on memory optimization for this prototype, so it's very possible that even training a small model like Qwen-3B can use more memory than what we normally use in SFT. We accept PRs!

# What's next
This is just a prototype to outline what a fully asynchronous RL training loop can look like. We wanted to build this directly in Tune to have a working example to compare to the sync implementation.

In the next phase of this project, we are going to factor out components into another library, and we are going to spend more time on API design to make sure we can craft something that people will love. We will post a public RFC when we are ready...

For the time being, please play with this prototype, tell us what you like, and most importantly tell us what we can be doing better!
