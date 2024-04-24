import os
from torch.utils.data import Dataset
from torchtune import utils
import deeplake
log = utils.get_logger("DEBUG")


class DeepLakeDataloader(Dataset):
    """
    A PyTorch Dataset class for loading data from ActiveLoop's DeepLake platform.

    This class serves as a data loader for working with datasets stored in ActiveLoop's DeepLake platform.
    It takes a DeepLake dataset object as input and provides functionality to load data from it
    using PyTorch's DataLoader interface.

    Attributes:
        ds (deeplake.Dataset): The dataset object obtained from ActiveLoop's DeepLake platform.

    Methods:
        __init__(self, ds): Initializes the DeepLakeDataloader with the given dataset object.
        __len__(self): Returns the number of samples in the dataset.
        __getitem__(self, idx): Retrieves a sample from the dataset at the specified index.

    Example:
        # Load a dataset from DeepLake and create a DataLoader
        dataset = load_deeplake_dataset("my_deep_lake_dataset")
        dataloader = DeepLakeDataloader(dataset)
    """

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        column_map = self.ds.tensors.keys()

        values_dataset = {}
        for el in column_map:  # {"column_name" : value}
            values_dataset[el] = self.ds[el][idx].text().astype(str)

        return values_dataset


def load_deep_lake_dataset(deep_lake_dataset: str, **config_kwargs) -> DeepLakeDataloader:
    """
    Load a dataset from ActiveLoop's DeepLake platform.

    This function loads a dataset from ActiveLoop's DeepLake platform using the provided dataset name.
    It sets up the necessary environment variables required for authentication and retrieves the dataset.
    If the environment variable ACTIVELOOP_TOKEN is not found, it prints a message indicating that only
    public datasets can be read.

    Args:
        deep_lake_dataset (str): The name of the dataset to load from DeepLake.

    Returns:
        DeepLakeDataloader: A data loader for the loaded dataset.

    Example:
        # Load a dataset named "my_deep_lake_dataset" from DeepLake
        dataloader = load_deeplake_dataset("my_deep_lake_dataset")
    """
    try:
        os.environ["ACTIVELOOP_TOKEN"] = os.getenv("ACTIVELOOP_TOKEN")
    except:
        log.info("ACTIVELOOP_TOKEN not found in environment variables only public datasets can be read.")

    ds = deeplake.dataset(deep_lake_dataset, **config_kwargs)
    log.info(f"Dataset loaded from deeplake: {ds}")
    return DeepLakeDataloader(ds)
