from torch.utils.data import Dataset

from .alpaca import AlpacaDataset


def get_dataset(name: str, split: str, **kwargs) -> Dataset:
    if name == "alpaca":
        return AlpacaDataset(split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
