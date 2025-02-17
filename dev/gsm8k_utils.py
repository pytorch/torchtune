import re
from tqdm import tqdm
import datasets


def gsm8k_get_answer(example):
    match = re.search(r"####\s*(\d+)", example['answer'])
    if match is not None:
        answer = int(match.group(1))
    else:
        answer = None
    return {'answer': answer}


def get_gsm8k_test(split='test'):
    gsm8k = datasets.load_dataset('openai/gsm8k', 'main')
    gsm8k = gsm8k[split].map(gsm8k_get_answer)
    gsm8k = gsm8k.filter(lambda x: x['answer'] is not None)
    return gsm8k


def process_tokenizer(tokenizer):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.chat_template = '{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'
    return tokenizer


def build_prompt(tokenizer, question, answer=None):
    # llama_prompt = lambda x: f"Given the following problem, reason and give a final answer to the problem.\nProblem: {x}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    llama_prompt = lambda x: x + " Let's think step by step."  # and output the final answer after \"####\"."
    messages = [{'role': 'user', 'content': llama_prompt(question)}]
    if answer is not None:
        messages.append({'role': 'assistant', 'content': answer})
    # if tokenizer.chat_template:
    #     prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=answer is None, add_special_tokens=True)
    # else:  # some default template
    prompt = f"{tokenizer.bos_token}Question: {llama_prompt(question).strip()}\nAnswer:"
    if answer is not None:
        prompt += f" {answer.strip()}{tokenizer.eos_token}"
    return prompt


def batch_generate_response(tokenizer, model, prompts, sampling_params={}):
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, padding_side='left')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    output = model.generate(**inputs, **sampling_params)
    responses = tokenizer.batch_decode(output, skip_special_tokens=True)
    return responses


def extract_answer(solution_str):
    matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", solution_str)
    matches = [m[0] if m[0] else m[1] for m in matches]
    final_answer = matches[-1] if matches else None
    try:
        final_answer = re.sub(r'[.,!?;:]+$', '', final_answer)  # remove trailing punctuation
        final_answer = int(float(final_answer.replace(',', '').replace('$', '')))
    except:
        final_answer = None

    return final_answer


def gsm8k_verifier(response, answer):
    model_answer = extract_answer(response)
    return 1.0 if model_answer == answer else 0


def score_maj_n(responses, answers, n):  # responses is (B*n, ), answers is (B,)
    print(len(responses), len(answers), n)
    assert len(responses) == len(answers) * n

    responses = [extract_answer(r) for r in responses]
    responses = [responses[i:i + n] for i in range(0, len(responses), n)]
    maj_responses = [max(set(r), key=r.count) for r in responses]

    pass_n = sum([a in r for a, r in zip(answers, responses)])
    maj_n = sum([a == r for a, r in zip(answers, maj_responses)])

    return pass_n / len(answers), maj_n / len(answers)


def evaluate_model(model, tokenizer, gsm8k, batch_size=1, verbose=False, n_samples=1, sampling_params={}):
    # build all prompts
    map_build_prompt = lambda x: {'prompt': build_prompt(tokenizer, x['question'])}
    gsm8k = gsm8k.map(map_build_prompt, num_proc=8)

    # inference
    if type(model).__module__.startswith('transformers'):
        responses = []
        for i in tqdm(range(0, len(gsm8k), batch_size)):
            batch = gsm8k.select(range(i, min(i + batch_size, len(gsm8k))))
            response = batch_generate_response(tokenizer, model, batch['prompt'], sampling_params=sampling_params)
            responses.extend(response)
    else:
        responses = model.generate(gsm8k['prompt'], **sampling_params)
        responses = [r_.text for r in responses for r_ in r.outputs]

    if verbose:
        for i, (response, prompt) in enumerate(zip(responses, gsm8k['prompt'])):
            print(f'Prompt: {prompt}')
            print(f'Response: {response}')
            print(f'Prediction: {extract_answer(response)}')
            print(f'Ground truth: {gsm8k[i // n_samples]["answer"]}')
            print()

    pass_n, maj_n = score_maj_n(responses, gsm8k['answer'], n_samples)
    print(f'Pass-{n_samples} accuracy: {pass_n:.4f}')
    print(f'Majority-{n_samples} accuracy: {maj_n:.4f}')

    return maj_n


def get_gsm8k_train_sft(tokenizer):  # contains the reasoning traces
    def process_gsm8k_sft(example):
        question = build_prompt(tokenizer, example['question'])

        # answer part
        numeric_answer = gsm8k_get_answer(example)['answer']
        answer = example['answer']
        id_delimiter = answer.rfind('####')
        if id_delimiter != -1:
            answer = answer[:id_delimiter].strip()
        answer += f"\nThe final answer is {numeric_answer}"
        target = build_prompt(tokenizer, example['question'], answer)
        target = target[len(question):]

        return {'inputs': question, 'targets': target}

    gsm8k_train = datasets.load_dataset('openai/gsm8k', 'main')['train']
    gsm8k_train = gsm8k_train.map(process_gsm8k_sft, num_proc=8, remove_columns=['question', 'answer'])
    return gsm8k_train


def get_gsm8k_train_rl(tokenizer, split='train'):  # only contains the final answer
    def process_gsm8k_rl(example):
        question = build_prompt(tokenizer, example['question'])
        answer = gsm8k_get_answer(example)['answer']
        return {'prompt': question, 'answer': answer}

    gsm8k_train = datasets.load_dataset('openai/gsm8k', 'main')[split]
    gsm8k_train = gsm8k_train.map(process_gsm8k_rl, num_proc=8, remove_columns=['question'])
    gsm8k_train = gsm8k_train.filter(lambda x: x['answer'] is not None)
    return gsm8k_train


if __name__ == '__main__':
    import argparse
    import torch
    import transformers

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--n_eval', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--vllm', action='store_true')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--n_samples', type=int, default=1)

    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer = process_tokenizer(tokenizer)

    if args.vllm:
        import vllm

        model = vllm.LLM(
            args.model_dir,
            max_model_len=1024,
            enable_prefix_caching=True,
            load_format='safetensors',
        )
        sample_params = {'max_tokens': 512, 'temperature': 0}
        if args.n_samples > 1:
            sample_params['n'] = args.n_samples
            sample_params['temperature'] = 0.7
        sample_params = vllm.SamplingParams(**sample_params)
        sample_kwargs = {'sampling_params': sample_params}
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=torch.bfloat16, device_map='auto'
        )
        sample_kwargs = {'max_length': 1024, 'do_sample': False, 'pad_token_id': tokenizer.eos_token_id}

    gsm8k = get_gsm8k_test(split=args.split).shuffle(seed=0)
    if args.n_eval is not None and args.n_eval < len(gsm8k):
        gsm8k = gsm8k.select(range(args.n_eval))

    accuracy = evaluate_model(model, tokenizer, gsm8k, batch_size=args.batch_size, verbose=args.verbose,
                              sampling_params=sample_kwargs,
                              n_samples=args.n_samples if args.vllm else 1)
    print(f'Accuracy: {accuracy:.4f}')