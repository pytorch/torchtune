from transformers import AutoModel, AutoTokenizer
import inspect

model = AutoModel.from_pretrained("/mnt/vast/share/inf2-training/models/open_source/Qwen2.5-VL-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("/mnt/vast/share/inf2-training/models/open_source/Qwen2.5-VL-7B-Instruct")

print(f"Model source file: {inspect.getfile(model.__class__)}")
input_ids = tokenizer("Hello, how are you?", return_tensors="pt")

output = model(**input_ids)

print(output)