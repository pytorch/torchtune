# TODOs on Ray recipe
0. Use tensordicts everywhere. Better yet, tensorclasses with the correct contracts.
1. Transition from Queue to torchrl.RayReplayBuffer: https://github.com/pytorch/rl/blob/main/examples/distributed/collectors/multi_nodes/ray_buffer_infra.py
2. We are not ready with a Grader abstraction, keep rewards as they are
3. The VLLMActor needs to connect to the LLMEnv in TorchRL.
4. Separately, we should have an example here to show how the env can manipulate a text env. For example, logic such that if the model didn't think for more than K tokens, add "But wait" at the end. Alternatively, there is a paper showing that you can put "Time's up" in there to stop after Q iterations, with Q that can be randomly picked per sample.
5. How to log metrics in VLLM? Tokens per second etc
6. Instantiate model from config the TorchTune way, currently hardcoded
7. The reference model needs to be hosted by another VLLM worker. Ditto for reward models. We need a good API esp for configs.
8. Trainer has epochs now, we need to move it away from this. Design is even simpler: we simply have a total number of iterations to respect and we keep sampling from the replay buffer. Logging gets more complex though because now you have to log how many times you have epoched over each prompt. This design is not trivial.
9. Bring over SFT memory optimization tricks: tiled CrossEntropy, activation offloading/checkpointing, torch.compile.
10. Reference model should run next to the scorer as it behaves like a Reward model: only one forward on an already-generated sequence.
11. Memory investigation: 3B memory usage is sus even for the fully-sync case where I was going OOM on H100s.
12.
