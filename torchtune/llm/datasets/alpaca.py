from typing import Callable, List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset


class AlpacaDataset(Dataset):
    """PyTorch Representation of Alpaca Dataset from Hugging Face.

    Args:
        split (str): Split to use.
        tokenizer (Callable): Tokenizer used to encode data.

    Example:
    {
        "instruction": "Create a classification task by clustering the given list of items.",
        "input": "Apples, oranges, bananas, strawberries, pineapples",
        "output": "Class 1: Apples, Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",
        "text": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a classification task by clustering the given list of items.\n\n### Input:\nApples, oranges, bananas, strawberries, pineapples\n\n### Response:\nClass 1: Apples, Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",
    }
    """

    def __init__(self, split: str, tokenizer: Callable) -> None:
        self.data = load_dataset("tatsu-lab/alpaca", split=split)
        self._tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self.transform(self.data["text"][index])

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
