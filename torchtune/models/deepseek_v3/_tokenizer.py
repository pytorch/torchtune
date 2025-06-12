from typing import Optional
from torchtune.modules.transforms.tokenizers import HuggingFaceBaseTokenizer, ModelTokenizer
from torchtune.modules.transforms import Transform


class DeepSeekV3Tokenizer(ModelTokenizer, Transform):

    def __init__(self,
                 path: str,
                 config_path: str,
                 max_seq_len: Optional[int] = None,
                 ) -> None:

        self.hf_tokenizer = HuggingFaceBaseTokenizer(path, tokenizer_config_json_path=config_path)
        self.max_seq_len = max_seq_len

    @property
    def vocab_size(self) -> int:
        return self.hf_tokenizer.get_vocab_size()
    
