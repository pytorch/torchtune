# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
import json
import os
import random

from functools import partial
from typing import Any, Dict

from tqdm import tqdm

import torch
from omegaconf import DictConfig

from torch import nn
from torch.utils.data import DataLoader

from torchtune import config, utils
import re

import numpy as np


logger = utils.get_logger("DEBUG")


def parse_the_result(output):
    pattern = r"\[\[(?:[^\]]|\][^\]])*\]\]"
    try:
         # Find all matches of the pattern in the output text
        matches = re.findall(pattern, output)
        return matches[-1]
    except:
        return output


def shuffle_demonstrations(text, rng):
    # Define the regex pattern to match the input-output pairs
    pattern = re.compile(r'(\[\[.*?\]\])\s*->\s*(\[\[.*?\]\])', re.DOTALL)

    # Find all matches in the text
    matches = pattern.findall(text)

    print(len(matches))

    # Process the matches to extract the input and output matrices
    input_output_pairs = [(match[0], match[1]) for match in matches]

    for i, pair in enumerate(input_output_pairs):
        # replace it with index
        text = text.replace(pair[0] + ' -> ' + pair[1], f"{i}-index")


    permutation_idxs = np.arange(len(input_output_pairs))
    # shuffle the indices
    rng.shuffle(permutation_idxs)

    for i, idx in enumerate(permutation_idxs):
        text = text.replace(f"{i}-index", input_output_pairs[idx][0] + ' -> ' + input_output_pairs[idx][1])

    return text




def parse_numpy_from_str(array_str):
    # Remove the surrounding brackets
    clean_str = array_str.replace('[', '').replace(']', '')

    # Split by whitespaces to get individual elements and convert to integers
    elements = list(map(int, clean_str.split()))

    # Determine shape of the array
    rows = array_str.count('\n') + 1
    cols = len(elements) // rows

    # Create the NumPy array
    array = np.array(elements).reshape((rows, cols)).astype(np.int8)

    return array

class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)

        utils.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)
        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        # merge adapter
        if utils.ADAPTER_KEY in ckpt_dict:
            for key in ckpt_dict[utils.ADAPTER_KEY].keys():
                ckpt_dict[utils.MODEL_KEY][key] = ckpt_dict[utils.ADAPTER_KEY][key]

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
        )

        # breakpoint()
        # hf_checkpointer = config.instantiate(cfg.checkpointer_to_save)
        # # save model in the format that can be loaded by transformers
        # # load
        # hf_checkpointer.load_checkpoint()
        # breakpoint()
        # ckpt_dict = {utils.MODEL_KEY: self._model.state_dict()}

        # hf_checkpointer.save_checkpoint(ckpt_dict, epoch=1)

        # breakpoint()

        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._ds, self._dataloader = self._setup_data(cfg_dataset=cfg.dataset)

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:

        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        with self._device:
            model.setup_caches(max_batch_size=1, dtype=self._dtype)

        return model

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
    ) -> DataLoader:
        """
        All data related setup happens here. Currently this recipe only supports
        Map-style Datasets which fit into memory and an option for random shuffling.
        Samplers, iterable datasets, and streaming datasets are not supported.
        """
        assert cfg_dataset.train_on_input == False, "Should be set to False for generation"

        ds = config.instantiate(
            cfg_dataset,
            tokenizer=self._tokenizer,
            train_on_input=False,
            split="test",
        )

        dataloader = DataLoader(
            dataset=ds,
            shuffle=False,
            batch_size=1,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=-100,
            ),
        )

        return ds, dataloader

    @torch.no_grad()
    def generate(self, cfg: DictConfig):
        outputs = []
        total = 0.0
        corrects = 0.0
        os.makedirs(cfg.output_path, exist_ok=True)
        rng = np.random.default_rng(cfg.seed)

        for idx, batch in tqdm(enumerate(self._dataloader)):
            input_ids, labels = batch
            filename = self._ds._data["idx"][idx]
            # basename
            filename = os.path.basename(filename)
            # replace jsonl
            filename = filename.replace(".jsonl", "")

            input_ids, labels = input_ids[0], labels[0]

            prompt = [token.item() for token, label in zip(input_ids, labels) if label == -100]

            target = [token.item() for token, label in zip(input_ids, labels) if label != -100]

            prompt = prompt[:-1] +  target[:3]

            target = target[3:-1]

            prompt_str = self._tokenizer.decode(prompt, truncate_at_eos=False)

            print("Prompt: ", prompt_str)
            breakpoint()

            target_str = self._tokenizer.decode(target, truncate_at_eos=False)

            prompt = torch.tensor(prompt, dtype=torch.int, device=self._device)

            custom_generate_next_token = None

            outputs_collection = []

            for iter in range(2):
                if iter != 0:
                    shuffled_prompt_str = shuffle_demonstrations(prompt_str, rng)
                    shuffled_prompt = self._tokenizer.encode_with_special_tokens(shuffled_prompt_str, add_bos=True, add_eos=False)
                    shuffled_prompt = torch.tensor(shuffled_prompt, dtype=torch.int, device=self._device)
                else:
                    shuffled_prompt = prompt
                    shuffled_prompt_str = prompt_str

                try:
                    generated_tokens = utils.generate(
                        model=self._model,
                        prompt=shuffled_prompt,
                        max_generated_tokens=8150 - len(prompt),
                        temperature=cfg.temperature if iter != 0 else 0.0,
                        top_k=cfg.top_k,
                        eos_id=self._tokenizer.eot_id,
                        custom_generate_next_token=custom_generate_next_token,
                    )
                except:
                    generated_tokens = prompt

                # print(self._tokenizer.decode(generated_tokens, truncate_at_eos=False))
                # breakpoint()


                generated_tokens = generated_tokens[len(prompt): len(prompt)+len(target)]

                output_str = self._tokenizer.decode(generated_tokens, truncate_at_eos=False)

                output_str = parse_the_result(output_str)

                outputs_collection.append(output_str)

            # try to cast it each to np array

            parsed_outputs = []
            for output in outputs_collection:
                try:
                    output = output.strip()
                    if output.endswith("]") and not output.endswith("]]"):
                        output = output + "]"
                    arr = parse_numpy_from_str(output)
                    parsed_outputs.append(arr)
                except:
                    continue

            if len(parsed_outputs) == 0:
                output_strs = ["", ""]
            else:
                # parsed_shapes = [arr.shape for arr in parsed_outputs]
                # # pick the most common output shape
                # output_shape = max(set(parsed_shapes), key=parsed_shapes.count)
                # # filter out the outputs that don't match the most common shape
                # parsed_outputs = [arr for arr in parsed_outputs if arr.shape == output_shape]

                # # for each row in the output, pick the most common value
                # selected_columns = []
                # for irow in range(output_shape[0]):
                #     columns = [tuple(arr[irow].tolist()) for arr in parsed_outputs]
                #     columns = max(set(columns), key=columns.count)
                #     selected_columns.append(columns)

                # output_str = str(np.array(selected_columns))
                output_strs = [str(arr) for arr in parsed_outputs]
                # sort by commonnes
                attempt1 = max(set(output_strs), key=output_strs.count)
                output_strs = [attempt for attempt in output_strs if attempt == attempt1]
                if len(output_strs) > 1:
                    attempt2 = max(set(output_strs), key=output_strs.count)
                else:
                    attempt2 = attempt1

                output_strs = [attempt1, attempt2]



            target_str = parse_the_result(target_str)


            if idx % 5 == 0:
                print("Generated output1: ", output_strs[0])
                print("Generated output2: ", output_strs[1])
                print("Target output: ", target_str)

            if target_str == "":
                label = False
            else:
                label = target_str in output_strs
            corrects += label
            total += 1
            outputs.append({"input": prompt_str, "target": target_str, "output": output_strs, "idx": filename})

            print("Accuracy: ", corrects/total)

            with open(os.path.join(cfg.output_path, "aresults.txt"), "a", encoding="utf-8") as label_file:
                print(f"{filename}\t{label}", file=label_file)

            with open(os.path.join(cfg.output_path, "results.jsonl"), "a", encoding="utf-8") as f:
                result = {
                    "idx": filename,
                    "input": prompt_str,
                    "target": target_str,
                    "pred": output_str,
                    "correct": label,
                }
                print(json.dumps(result), file=f)

        # print a done.txt file
        with open(os.path.join(cfg.output_path, "done.txt"), "w") as f:
            print(f"Done!, accuracy: {corrects/total}", file=f)





@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
