import itertools
from torchtune.models.llama2 import llama2, llama2_tokenizer
from torchtune import utils
from torchtune.modules import rlhf
import torch
import time
import sys

args = sys.argv

torch.manual_seed(42)
device = torch.device("cuda")
dtype = torch.float32
with utils.set_default_dtype(dtype), device:
    model = llama2(
        vocab_size=32000, num_layers=22, num_heads=32, embed_dim=2048, max_seq_len=2048, norm_eps=1e-5, num_kv_heads=4
    )
utils.validate_expected_param_dtype(model.named_parameters(), dtype=dtype)
checkpointer = utils.FullModelHFCheckpointer(
    checkpoint_dir="./target/1b_normal/",
    checkpoint_files=[
        "pytorch_model.bin",
    ],
    model_type="LLAMA2",
    output_dir="./tmp/",
)

state_dict = checkpointer.load_checkpoint()["model"]
model.load_state_dict(state_dict=state_dict)
model.eval()
tokenizer = llama2_tokenizer("./target/1b_normal/tokenizer.model")

prompt = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla fermentum sapien vitae arcu volutpat egestas. Duis euismod nunc lorem, aliquet laoreet ex blandit vel. Vivamus sodales lacus velit, a hendrerit erat ornare vitae. Nam auctor nulla sit amet tempus pretium. Ut bibendum ullamcorper elit vel porttitor. Maecenas ut lacus in sapien suscipit condimentum. Sed eu massa mauris. Nullam id diam erat. Aenean quis nunc eu libero ultrices ullamcorper ut placerat odio. Aenean nunc elit, tincidunt in erat non, congue finibus tellus. Donec tellus nibh, imperdiet non justo nec, rutrum efficitur lacus.

Ut non commodo nisi. Nulla dapibus porta velit facilisis gravida. Phasellus interdum turpis quis facilisis ornare. Praesent maximus mauris eget quam rutrum aliquam. Duis vitae libero in tellus maximus gravida sed in augue. Aenean eu efficitur quam, id hendrerit velit. Donec porta sit amet mauris ut auctor. Suspendisse potenti. Maecenas pulvinar elit a enim volutpat, commodo pulvinar nisi rhoncus. 
"""

# prompt = "The quick brown fox jumped over the lazy dog and went to"

batch_size = 2
max_generated_tokens = 256
with device:
    model.setup_caches(batch_size=batch_size, dtype=dtype)

prompt = torch.tensor(tokenizer.encode(prompt, add_eos=False), dtype=torch.int, device=device).repeat(batch_size, 1)
# prompt = torch.hstack(
#     (torch.ones(batch_size, prompt.shape[-1] // 4, device=device, dtype=torch.int) * tokenizer.pad_id, prompt)
# )
# prompt[:2][: prompt.shape[-1] // 4] = tokenizer.pad_id

# ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']


def generate_with_compile():
    custom_generate_next_token = torch.compile(utils.generate_next_token, fullgraph=True, backend="inductor")
    print("Warmup run!")
    t0 = time.perf_counter()
    utils.generate(
        model=model,
        prompt=prompt,
        max_generated_tokens=2,
        temperature=1.0,
        top_k=None,
        stop_tokens=None,
        custom_generate_next_token=custom_generate_next_token,
    )
    t = time.perf_counter() - t0
    print(f"Time for warmup: {t:.02f} sec")

    t0 = time.perf_counter()
    outputs = utils.generate(
        model=model,
        prompt=prompt,
        max_generated_tokens=max_generated_tokens,
        temperature=1.0,
        top_k=None,
        stop_tokens=None,
        custom_generate_next_token=custom_generate_next_token,
    )
    t = time.perf_counter() - t0
    return outputs, t


def generate():
    t0 = time.perf_counter()
    outputs = utils.generate(
        model=model,
        prompt=prompt,
        max_generated_tokens=max_generated_tokens,
        temperature=1.0,
        top_k=None,
        stop_tokens=None,
        custom_generate_next_token=None,
    )
    t = time.perf_counter() - t0
    return outputs, t


def generate_rlhf():
    t0 = time.perf_counter()
    generated_tokens, _ = rlhf.generate_with_logits(
        model=model,
        prompt=prompt,
        max_generated_tokens=max_generated_tokens,
        temperature=1.0,
        top_k=None,
    )
    t = time.perf_counter() - t0
    return generated_tokens.tolist(), t


with torch.no_grad():
    if len(sys.argv) > 1:
        if sys.argv[1] == "compile":
            generated_tokens, t = generate_with_compile()
        elif sys.argv[1] == "rlhf":
            generated_tokens, t = generate_rlhf()
    else:
        generated_tokens, t = generate()

print(f"Output:\n {tokenizer.decode(generated_tokens[0])}")
tokens_generated = len(generated_tokens[0]) - prompt.size(0)
tokens_sec = tokens_generated / t
model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(model.parameters(), model.buffers())])

print(f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
