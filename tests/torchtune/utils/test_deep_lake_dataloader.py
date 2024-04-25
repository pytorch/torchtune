from unittest.mock import patch
from torchtune.datasets import DeepLakeDataloader, load_deep_lake_dataset
import deeplake


class TestDeepLakeDataloader:
    @patch("deeplake.dataset")
    def test_init(self, mock_dataset):
        dl = DeepLakeDataloader(mock_dataset)
        assert dl.ds == mock_dataset

    def test_len(self):
        ds = deeplake.dataset("test1", overwrite=True)
        ds.create_tensor("id", htype="text", exist_ok=True)
        ds.create_tensor("type", htype="text", exist_ok=True)
        ds.append({
            "id": "id1",
            "type": "type1",
        })
        ds.append({
            "id": "id2",
            "type": "type2",
        })
        dl = DeepLakeDataloader(ds)
        assert len(dl) == 2

    def test_get_item(self):
        ds = deeplake.dataset("test", overwrite=True)
        ds.create_tensor("id", htype="text", exist_ok=True)
        ds.create_tensor("type", htype="text", exist_ok=True)
        ds.append({
            "id": "id1",
            "type": "type1",
        })
        ds.append({
            "id": "id2",
            "type": "type2",
        })
        dl = DeepLakeDataloader(ds)
        assert dl[0] == {'id': 'id1', 'type': 'type1'}
        assert dl[1] == {'id': 'id2', 'type': 'type2'}


def test_load_deep_lake_dataset():
    with patch("deeplake.dataset") as mock_dataset:
        fake_ds = deeplake.dataset()
        mock_dataset.return_value = fake_ds
        dl = load_deep_lake_dataset("test", overwrite=True)
        assert isinstance(dl, DeepLakeDataloader)
        assert dl.ds == fake_ds
