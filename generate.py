from torchtune.models.llama2 import llama2, llama2_tokenizer
from torchtune import utils

model = llama2(
    vocab_size=32000, num_layers=22, num_heads=32, embed_dim=2048, max_seq_len=2048, norm_eps=1e-5, num_kv_heads=4
)
checkpointer = utils.FullModelHFCheckpointer(
    checkpoint_dir="./dummy",
    checkpoint_files=[
        "pytorch_model.bin",
    ],
    model_type="LLAMA2",
    output_dir="./tmp/",
)

state_dict = checkpointer.load_checkpoint()["model"]
model.load_state_dict(state_dict=state_dict)
tokenizer = llama2_tokenizer("./dummy/tokenizer.model")