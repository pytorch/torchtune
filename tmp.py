import json

from transformers import AutoTokenizer
from transformers.utils import get_json_schema

from torchtune.models.qwen2 import qwen_tokenizer
from torchtune.data import Message


hf = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# tt = qwen_tokenizer(
#     qwen_version='2.5', 
#     path='/tmp/Qwen2.5-0.5B-Instruct/vocab.json', 
#     merges_file='/tmp/Qwen2.5-0.5B-Instruct/merges.txt', 
#     tools=[json.dumps(get_json_schema(tool)) for tool in tools] if tools is not None else None,
# )


def test(messages, tools=None):
    tt = qwen_tokenizer(
        qwen_version='2.5', 
        path='/tmp/Qwen2.5-0.5B-Instruct/vocab.json', 
        merges_file='/tmp/Qwen2.5-0.5B-Instruct/merges.txt', 
        tools=[json.dumps(get_json_schema(tool)) for tool in tools] if tools is not None else None,
    )

    a = hf([hf.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True
    )], return_tensors="pt")
    print(type(a), a)

    print('\n\n\n')
    messages.append({'role': 'assistant', 'content': ''})
    b = tt.tokenize_messages([Message(x['role'], x['content']) for x in messages], add_eos=False)
    print(type(b), b)

    print('\n\n\n')
    print(len(a['input_ids'][0]), len(b[0]))
    for x, y in zip(a['input_ids'][0], b[0]):
        print('O' if x.item() == y else 'X', x.item(), y)
        assert x.item() == y

test([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Give me a short introduction to large language model."},
])


test([
    {"role": "user", "content": "Give me a short introduction to large language model."},
])


def get_current_temperature(location: str):
    """
    Gets the temperature at a given location.

    Args:
        location: The location to get the temperature for
    """
    return 22.0  # bug: Sometimes the temperature is not 22. low priority

test([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Give me a short introduction to large language model."},
], [get_current_temperature])

test([
    {"role": "user", "content": "Give me a short introduction to large language model."},
], [get_current_temperature])
