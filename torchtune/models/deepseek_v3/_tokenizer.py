from typing import Optional
from torchtune.modules.transforms.tokenizers import HuggingFaceBaseTokenizer, ModelTokenizer
from torchtune.modules.transforms import Transform
from functools import cached_property

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
    
    def encode(self, *args, **kwargs) -> list[int]:
        return self.hf_tokenizer.encode(*args, **kwargs)
    
    def decode(self, *args, **kwargs) -> str:
        return self.hf_tokenizer.decode(*args, **kwargs)
    
    @property
    def bos_id(self) -> int:
        return self.hf_tokenizer.bos_id
    
    @property
    def eos_id(self) -> int:
        return self.hf_tokenizer.eos_id
    
    @cached_property
    def pad_id(self) -> int:
        return self.hf_tokenizer.tokenizer.token_to_id(self.hf_tokenizer.config.get("pad_token"))
    

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("smohammadi/deepseek-v3-micro")
    text = "Hello, world!"
    tokens = tokenizer.encode(text, add_special_tokens=True)
    print(tokens)
    print(tokenizer.decode(tokens))

    tt_tokenizer = DeepSeekV3Tokenizer(
        path="/Users/salmanmohammadi/projects/torchtune/target/scripts/deepseek/deepseek-v3-micro/tokenizer.json",
        config_path="/Users/salmanmohammadi/projects/torchtune/target/scripts/deepseek/deepseek-v3-micro/tokenizer_config.json",
        max_seq_len=1024
    )
    tt_tokens = tt_tokenizer.encode(text, add_bos=True, add_eos=True)
    print(tt_tokens)
    print(tt_tokenizer.decode(tt_tokens))
    import ipdb; ipdb.set_trace()