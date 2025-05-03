# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from tensordict import from_dataclass, lazy_stack, TensorClass  # noqa
from tensordict.utils import _zip_strict

from .vllm_completion_output import VllmCompletionOutput


class RequestOutput(TensorClass["nocast"]):
    request_id: str
    prompt: str
    prompt_token_ids: str
    prompt_logprobs: str
    outputs: str
    finished: str
    metrics: str
    lora_request: str
    encoder_prompt: str
    encoder_prompt_token_ids: str
    num_cached_tokens: str

    def __post_init__(self):
        def get_logprob(output):
            t = []
            for v, tid in zip(output.logprobs, output.token_ids):
                t.append(
                    v[tid]["logprob"] if v[tid].get("logprob") is not None else 1.0
                )
            return torch.tensor(t)

        def postproc(output):
            if output.logprobs:
                output.logprobs = get_logprob(output)
            output.token_ids = torch.tensor(output.token_ids)
            return output

        if isinstance(self.outputs, list):
            outputs = self.outputs
            outputs = [
                postproc(from_dataclass(output, dest_cls=VllmCompletionOutput))
                for output in outputs
            ]
            if len(outputs) == 1:
                self.outputs = outputs[0]
            else:
                raise NotImplementedError(
                    "Calling self.outputs = maybe_dense_stack(outputs) which is never defined!"
                )

            if self.prompt_logprobs is not None:
                self.prompt_logprobs = torch.tensor(
                    [
                        v[tid].logprob if v is not None else 0.0
                        for v, tid in _zip_strict(
                            self.prompt_logprobs, self.prompt_token_ids
                        )
                    ]
                )
            self.prompt_token_ids = torch.tensor(self.prompt_token_ids)
            self.num_cached_tokens = torch.tensor(self.num_cached_tokens)

    @classmethod
    def from_request_output(cls, requests):
        out = lazy_stack(
            [
                cls(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    prompt_token_ids=request.prompt_token_ids,
                    prompt_logprobs=request.prompt_logprobs,
                    outputs=request.outputs,
                    finished=request.finished,
                    metrics=request.metrics,
                    lora_request=request.lora_request,
                    encoder_prompt=request.encoder_prompt,
                    encoder_prompt_token_ids=request.encoder_prompt_token_ids,
                    num_cached_tokens=request.num_cached_tokens,
                )
                for request in requests
            ]
        )
        return out
