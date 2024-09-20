import pytest

from tests.test_utils import DummyTokenizer
from torchtune.data import Message

from torchtune.modules.tokenizers import tokenize_messages_no_special_tokens

class TestTokenizerUtils:
    @pytest.fixture
    def tokenizer(self):
        return DummyTokenizer(max_seq_len=100)
    
    @pytest.fixture
    def messages(self):
        return [
            Message(role="user", content="hello world!"),
            Message(role="assistant", content="hello back!"),
        ]

    @pytest.mark.parametrize(
        "add_bos, add_eos",
        [
            (True, True),
            (False, False),
        ],
    )
    def test_tokenize_no_special_tokens(self, tokenizer, messages, add_bos, add_eos):
        tokens, mask = tokenize_messages_no_special_tokens(
            tokenizer,
            messages,
            bos_id=tokenizer.bos_id if add_bos else None,
            eos_id=tokenizer.eos_id if add_eos else None,
        )

        assert len(tokens) == len(mask)
        
        if add_bos:
            assert tokens[0] == tokenizer.bos_id
            assert mask[0] == True
        else:
            assert tokens[0] != tokenizer.bos_id
            assert mask[0] == False
        
        if add_eos:
            assert tokens[-1] == tokenizer.eos_id
            assert mask[-1] == True
        else:
            assert tokens[-1] != tokenizer.eos_id
            assert mask[-1] == False
