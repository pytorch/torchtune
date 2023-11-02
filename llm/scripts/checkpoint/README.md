### Checkpoint conversion to native PyTorch format

The original Llama-2 models were trained using a different implementation (for example, non-fused QKV matrix, no SDPA) as well as [fairscale](https://github.com/facebookresearch/fairscale) based
paralellism.

TorchTBD seeks to leverage the power of native PyTorch including PyTorch-native distributed APIs such as FSDP and Tensor Parallelism. To train Llama-2 models using TorchTBD, checkpoints must be
converted as a result.


#### Offline conversion of original checkpoint

Assuming your original checkpoint lives at `path_to_native_checkpoint`, invoking `python -m llm.scripts.checkpoint.convert_to_native --checkpoint_path <path_to_native_checkpoint>` from your `torch_tbd` root will write out a native checkpoint to
`/tmp/native_checkpoints/llama2-{x}b` where x is the model size. Currently, only the 7b parameter model is supported.

NOTE: This checkpoint conversion is under heavy development and expected to change. For example, we will onboard to `torch.distributed.checkpoint` and manage checkpoints at the directory, instead of
file, level.
