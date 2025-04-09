from typing import Any, Callable, Dict, Mapping, Optional

from torchtune.data import Message
from torchtune.datasets._preference import PreferenceDataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform


import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import ChosenRejectedToMessages, CROSS_ENTROPY_IGNORE_IDX

from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform
import random, json
import os
from torchtune.data import InputOutputToMessages
from torchtune import config, generation, modules, rlhf, training




class GRPO_Dataset(Dataset):
    def __init__(
        self,
        *,
        source: str,
        message_transform: Transform,
        tokenizer: ModelTokenizer,
        ref_model=None,
        temperature= 0.5, 
        filter_fn: Optional[Callable] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._message_transform = message_transform
        self._data = load_dataset(source, **load_dataset_kwargs)
        self.ref_model = ref_model
        self._temperature = temperature
        # Pre-compute and move stop_token_ids to GPU once instead of every __getitem__ call
        self._stop_token_ids = torch.tensor(self._tokenizer.stop_tokens, device="cuda")

        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)
            
        # Pre-allocate buffers for commonly used tensors
        self._device = torch.device("cuda")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int):
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        # Transform sample first
        transformed_sample = self._message_transform(sample)
        
        # Tokenize the messages
        tokenized_dict = self._tokenizer(transformed_sample)
        
        query_response_tokens = np.array(tokenized_dict["tokens"])
        mask = np.array(tokenized_dict["mask"])
        
        labels = np.where(
            tokenized_dict["mask"],
            CROSS_ENTROPY_IGNORE_IDX,
            tokenized_dict["tokens"],
        )
        
        response_tokens = torch.tensor(
            labels[labels != -100], 
            dtype=torch.long,
            device=self._device
        ).unsqueeze(0)
        
        # Use your provided logic to extract query tokens
        query_tokens = query_response_tokens[mask][:-1]
        
        # Create query_response_tokens directly as tensors
        query_response_tokens_tensor = torch.tensor(
            query_response_tokens,
            dtype=torch.long,
            device=self._device
        ).unsqueeze(0)
        
        # Combine logic for logprobs without intermediate lists
        
        # Create padding masks efficiently
        query_response_padding_masks = query_response_tokens_tensor != self._tokenizer.pad_id
        
        # Generate position IDs from padding mask
        position_ids = generation.get_position_ids_from_padding_mask(query_response_padding_masks)
        
        # Create attention masks from the padding mask
        masks = generation.get_causal_mask_from_padding_mask(query_response_padding_masks)
        
        # Compute ref_logprobs using the reference model
        with torch.no_grad():
            ref_logits = self.ref_model(
                query_response_tokens_tensor, 
                input_pos=position_ids, 
                mask=masks
            )
        ref_logits = rlhf.truncate_sequence_for_logprobs(ref_logits, len(query_tokens))
            
        ref_logprobs = rlhf.batched_logits_to_logprobs(
            ref_logits, 
            response_tokens, 
            self._temperature
        )

        response_logprobs=ref_logprobs
        
        # Truncate sequences at the first stop token
        response_padding_masks, response_tokens = rlhf.truncate_sequence_at_first_stop_token(
            response_tokens, 
            self._stop_token_ids, 
            self._tokenizer.pad_id
        )
        
        # Get advantages directly from sample
        advantages = sample["advantage"]
        
        # Package everything into a trajectory dictionary - convert logprobs to tensor
        trajectory = {
            "query_responses": query_response_tokens_tensor,
            "position_ids": position_ids,
            "masks": masks,
            "logprobs": torch.tensor(response_logprobs, device=self._device).unsqueeze(0),
            "ref_logprobs": ref_logprobs,
            "advantages": torch.tensor(advantages, device=self._device).unsqueeze(0),
            "response_padding_masks": response_padding_masks,
            "query_len": len(query_tokens),
            "type": torch.tensor(sample["completion_reward"]),
            "response_tokens": response_tokens
        }
        
        return trajectory



def grpo_dataset(
    tokenizer: ModelTokenizer,
    ref_model, 
    *,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    new_system_prompt: Optional[str] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> PreferenceDataset:
    column_map = column_map or {
        "input": "prompt",
        "output": "output",
    }


    message_transform=InputOutputToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )

    return GRPO_Dataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        ref_model=ref_model,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )