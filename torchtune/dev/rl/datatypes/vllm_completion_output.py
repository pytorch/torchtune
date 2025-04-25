import vllm
from tensordict import from_dataclass

VllmCompletionOutput = from_dataclass(vllm.outputs.CompletionOutput)
