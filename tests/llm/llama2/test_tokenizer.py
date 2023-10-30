import pytest
from llm.llama2.tokenizer import Tokenizer
from pathlib import Path
import torch

ASSETS = Path(__file__).parent.parent.parent / "assets"

class TestTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_file(str(ASSETS / "m.model"))

    def test_encode(self, tokenizer):
        assert tokenizer.encode("Hello world!") == [1, 12, 1803, 1024, 103, 2]
        assert tokenizer.encode("Hello world!", add_eos=False) == [1, 12, 1803, 1024, 103]
        assert tokenizer.encode("Hello world!", add_bos=False) == [12, 1803, 1024, 103, 2]
        assert tokenizer.encode("Hello world!", add_eos=False, add_bos=False) == [12, 1803, 1024, 103]
        assert tokenizer.encode(["Hello world!"]) == [[1, 12, 1803, 1024, 103, 2]]
        assert torch.allclose(tokenizer.encode("Hello world!", return_as_tensor_with_dtype=torch.long), torch.tensor([1, 12, 1803, 1024, 103, 2]))
        print(tokenizer.encode(["Hello world!", "what"], return_as_tensor_with_dtype=torch.long))
        assert torch.allclose(tokenizer.encode(["Hello world!", "what"], return_as_tensor_with_dtype=torch.long), torch.tensor([1, 12, 1803, 1024, 103, 2]))

    # def test_decode(self, tokenizer):
    #     assert tokenizer.decode([1, 12, 1803, 1024, 103, 2]) == "Hello world!"
    #     assert tokenizer.decode([[1, 12, 1803, 1024, 103, 2]]) == ["Hello world!"]
    #     assert tokenizer.decode([1, 12, 1803, 1024, 103, 2],
