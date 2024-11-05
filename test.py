# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: (philip) remove after tests

from transformers import AutoModelForCausalLM  # , AutoTokenizer

model_id = "meta-llama/Llama-3.2-11B-Vision"
peft_model_id = "/tmp/Llama-3.2-11B-Vision-Instruct/"

model = AutoModelForCausalLM.from_pretrained(model_id)
model.load_adapter(peft_model_id)
