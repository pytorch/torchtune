### Checkpoint conversion to native PyTorch format

The original [Llama-2 models](https://github.com/facebookresearch/llama/blob/main/llama/model.py#L413) were trained using a different implementation (for example, non-fused QKV matrix, no SDPA) as well as [fairscale](https://github.com/facebookresearch/fairscale) based
parallelism.

TorchTune seeks to leverage the power of native PyTorch including PyTorch-native distributed APIs such as FSDP and Tensor Parallelism. To train Llama-2 models using TorchTune, checkpoints must be converted as a result.

You can dowload the original Llama 2 weights by
1. Getting approval from this form https://ai.meta.com/llama/
2. Make sure you've been granted access here https://huggingface.co/meta-llama/Llama-2-7b
3. Make sure you've set your access token as an environment variable https://huggingface.co/docs/hub/security-tokens

Then once the approvals are ready you can download the weights from HuggingFace

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="meta-llama/Llama-2-7b")
```

#### Offline conversion of original checkpoint

Assuming your original checkpoint lives at `path_to_original_checkpoint`, invoking the below commands will write out a native checkpoint to
`/tmp/native_checkpoints/llama2-{x}b` where x is the model size. Currently, only the 7b parameter model is supported.

```bash
cd torchtune
python -m scripts.llama2_checkpoint.convert_llama2_to_native --checkpoint_path <path_to_original_checkpoint> --device cuda:0
```

NOTE: This checkpoint conversion is under heavy development and expected to change. For example, we will onboard to `torch.distributed.checkpoint` and manage checkpoints at the directory, instead of file, level.
