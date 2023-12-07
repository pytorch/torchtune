from torch.utils.data import Dataset

from .alpaca import AlpacaDataset


def get_dataset(name: str, **kwargs) -> Dataset:
    if name == "alpaca":
        return AlpacaDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
