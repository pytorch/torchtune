import pytest
from torchtune.modules.transforms.tokenizers import HFTokenizer

# TODO: change this (just for testing)
TOKENIZER_DIR = "/data/users/ebs/phi4/"


class TestHFTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return HFTokenizer(
            path=TOKENIZER_DIR + "tokenizer.json",
            config_path=TOKENIZER_DIR + "tokenizer_config.json",
        )

    def test_tokenizer(self, tokenizer):
        import pdb

        pdb.set_trace()
        raise ValueError("done")
