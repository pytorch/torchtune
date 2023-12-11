from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

# Not ideal to import this type here but it's needed for the transform function
from tune.blocks.models.llama2.tokenizer import Tokenizer


class AlpacaDataset(Dataset):
    """PyTorch Representation of the Alpaca Dataset from Hugging Face.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.

    Data input format:
    {
        "instruction": "Create a classification task by clustering the given list of items.",
        "input": "Apples, oranges, bananas, strawberries, pineapples",
        "output": "Class 1: Apples, Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",
        "text": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a classification task by clustering the given list of items.\n\n### Input:\nApples, oranges, bananas, strawberries, pineapples\n\n### Response:\nClass 1: Apples, Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",
    }

    Example:
    >>> alpaca_ds = AlpacaDataset(tokenizer=tokenizer)
    >>> for batch in Dataloader(alpaca_ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """

    def __init__(self, tokenizer: Tokenizer, **kwargs) -> None:
        self._data = load_dataset("tatsu-lab/alpaca", split="train")
        self._tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self.transform(self._data[index]["text"])

    def transform(self, sample: str) -> Tuple[List[int], List[int]]:
        """Split a sample on 'response' tag to create input and labels.

        Args:
            sample (str): Sample text.

        Returns:
            Tuple of encoded inputs and labels.
        """
        response_tag = "\n\n### Response:\n"
        split_text = sample.split(response_tag)
        instructions_and_inputs = self._tokenizer.encode(split_text[0] + response_tag)
        labels = self._tokenizer.encode(split_text[1])
        return instructions_and_inputs, labels
