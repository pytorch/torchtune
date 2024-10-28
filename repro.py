# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Demonstrate OffloadActivations NaN gradients from tensor deletion data race.

To run:

pip install torch==2.5.0 torchtune@git+https://github.com/pytorch/torchtune.git@main torchao==0.5.0 transformers liger_kernel
CUDA_VISIBLE_DEVICES=0 python offload_activations_nan.py
"""

import contextlib
import functools

import liger_kernel.transformers
import torch
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper
import torch.distributed.fsdp.wrap
import torch.optim
import torchtune.training
import transformers
import transformers.models.llama.modeling_llama
from torch import nn

# torch.autograd.set_detect_anomaly(True)

BATCH_SIZE = 2
SEQUENCE_LENGTH = 2048
EMBEDDING_DIM = 4096


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_DIM, dtype=torch.float32)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mask = torch.randint(
            0,
            2,
            (BATCH_SIZE, SEQUENCE_LENGTH, 1),
            dtype=torch.bool,
            device=self.weight.device,
        )
        return (
            input.to(torch.float32) * mask
            + (torch.randn_like(self.weight) * self.weight * ~mask)
        ).to(input)


def main() -> None:
    liger_kernel.transformers.apply_liger_kernel_to_llama()

    torch.cuda.set_device(device := torch.device("cuda:0"))
    torch.set_default_dtype(dtype := torch.bfloat16)

    with device:
        config = transformers.LlamaConfig(num_hidden_layers=1)
        llama_model = transformers.LlamaForCausalLM(config).eval()
        for param in llama_model.parameters():
            param.requires_grad = False

        my_model = MyModel()

    auto_wrap_policy = functools.partial(
        torch.distributed.fsdp.wrap.transformer_auto_wrap_policy,
        transformer_layer_cls={
            transformers.models.llama.modeling_llama.LlamaDecoderLayer
        },
    )
    torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing(
        llama_model,
        checkpoint_wrapper_fn=torch.distributed.algorithms._checkpoint.checkpoint_wrapper.checkpoint_wrapper,
        auto_wrap_policy=auto_wrap_policy,
    )

    optimizer = torch.optim.AdamW(params=my_model.parameters())

    vocab_size = llama_model.config.vocab_size
    input_ids = torch.randint(
        low=0, high=vocab_size, size=(BATCH_SIZE, SEQUENCE_LENGTH), device=device
    )

    for i in range(1, 0, -1):
        with (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
            )
            if True  # i == 0
            else contextlib.nullcontext()
        ) as profiler:

            # import logging

            if True:  # i == 0:
                torch.cuda.memory._record_memory_history()

            from torch.utils._python_dispatch import TorchDispatchMode

            class LoggingMode(TorchDispatchMode):
                def __init__(self, logger):
                    self.logger = logger
                    return super().__init__()

                def contents(self, t):
                    return (
                        (t.shape, hex(t.untyped_storage().data_ptr()))
                        if isinstance(t, torch.Tensor)
                        else t
                    )

                def __torch_dispatch__(self, func, types, args=(), kwargs=()):
                    outputs = func(*args, **kwargs)
                    if isinstance(outputs, torch.Tensor) or outputs is None:
                        outputs = [outputs]
                    print(
                        f"{func.__name__}, "
                        f"args: {[self.contents(a) for a in args]}, "
                        f"kwargs: {[self.contents(a) for a in kwargs]}, "
                        f"outputs: {[self.contents(o) for o in outputs]}"
                    )
                    return func(*args, **kwargs)

            # with LoggingMode(logging.Logger("torch")):
            with torchtune.training.OffloadActivations(use_streams=True):
                output = llama_model(
                    inputs_embeds=my_model(
                        llama_model.get_input_embeddings()(input_ids)
                    ),
                    labels=input_ids,
                    use_cache=False,
                )

            with torch.autograd.set_multithreading_enabled(False):
                output.loss.backward()

            if (
                True
            ):  # i != 0:  # this produces weird trace/snapshot artifacts, so skip it
                grad = my_model.weight.grad
                print(grad)
                print(f"{i=}: {grad.isnan().any().item()=}")
                print(f"{i=}: {torch.equal(grad, torch.zeros_like(grad))=}")

        optimizer.zero_grad(set_to_none=True)

    assert profiler is not None
    profiler.export_chrome_trace("./offload_activations.json")
    torch.cuda.memory._dump_snapshot("offload_activations_nan_snapshot.pkl")

    print(f"max_memory_allocated={torch.cuda.max_memory_allocated() / 2 ** 30} GiB")
    print("done.")


if __name__ == "__main__":
    main()
