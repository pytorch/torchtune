import argparse
import ast
import datasets


def clean_rule(rule):
    # Looking for 1. or 2. etc.
    splits = rule.split(". ", 1)
    if len(splits) > 1:
        rule = splits[1].strip()
    else:
        rule = rule.strip()
    return rule


def clean_explanation(explanation):
    # Looking for "Turn x: "
    explanation = explanation.split(": ", 1)[1].strip()
    return explanation

def parse_string_list_problem(string_list):
    # Format: "1. ['Turn 1. ...', 'Turn 2. ...', ...]\n"
    string_list = string_list.split(". ", 1)[1].strip()
    turn_count = string_list.count("Turn ")
    native_list = ast.literal_eval(string_list)
    if len(native_list) > turn_count:
        native_list = string_list.split("Turn ")[1:]
    return native_list


def parse_string_list(string_list):
    # Format: "1. ['PASS', 'PASS', 'PASS']\n"
    string_list = string_list.split(". ", 1)[1].strip()
    native_list = ast.literal_eval(string_list)
    return native_list


def preprocess_dataset(dataset_path, subset=None, split=None, size=None, local=False):
    if local:
        dataset = datasets.load_dataset('json', data_files=dataset_path)['train']
    else:
        dataset = datasets.load_dataset(dataset_path, subset, split=split)
    print(f"Examples in {subset} {split}: {len(dataset)}")

    examples = []
    for row in dataset:
        # Get rules
        rules = row['rules']
        # Get turns
        dialogue = row['dialogue']
        delimiter = "'User':"
        dialogue_turns = [f"{delimiter}{item}" for item in dialogue.split(delimiter) if item]
        # Get discussions, explanations, and labels
        discussions = row['discussions'] # List of strings
        explanations = row['explanations'] # List of strings
        
        labels = row['labels'] # List of strings

        for i, rule in enumerate(rules):
            for j in range(len(dialogue_turns)):
                example = {}
                dialogue_subset = "".join(dialogue_turns[:j+1])
                # Construct input
                try:
                    rule = clean_rule(rule)
                except Exception as e:
                    print(f"BAD RULE: {rule}")
                    raise e
                example['input'] = f'''
Rule Agent must follow:
{rule}

Conversation:
{dialogue_subset}
'''
            # Construct ouput
            try:
                if subset == "hard" and split == "test":
                    discussion = parse_string_list_problem(discussions[i])[j]
                else:
                    discussion = parse_string_list(discussions[i])[j] # Starts with "Turn x: "
                discussion = clean_explanation(discussion)
            except Exception as e:
                print(f"BAD DISCUSSION: {discussions}")
                raise e
            
            try:
                if subset == "hard" and split == "test":
                    explanation = parse_string_list_problem(explanations[i])[j]
                else:
                    explanation = parse_string_list(explanations[i])[j] # Starts with "Turn x: "
                explanation = clean_explanation(explanation)
            except Exception as e:
                print(f"BAD EXPLANATION: {explanations}")
                raise e
            
            label = parse_string_list(labels[i])[j]
            example['output'] = f"{discussion} {explanation} Compliance output: {label}"
            examples.append(example)


    torchtune_dataset = datasets.Dataset.from_list(examples)
    torchtune_dataset = torchtune_dataset.shuffle(seed=42)

    if size is None or len(torchtune_dataset) < size:
        size = len(torchtune_dataset)
    torchtune_dataset = torchtune_dataset.select(range(size))

    file_path = f"data/{subset}_{split}_{size}.jsonl"

    torchtune_dataset.to_json(file_path, orient='records', lines=True, indent=None)
    print(f"Examples in dataset preprocessed for TorchTune: {size}")
    print(f"Saved to {file_path}")


def main(args):
    # Local:
    # preprocess_dataset("test.jsonl", local=True)

    huggingface_dataset = "tomg-group-umd/compliance"
    # Subset choices are "easy" or "hard"
    # Easy: 9007 train, 1793 val, 67 test
    # Hard: 1670 train, 313 val, 44 test
    subsets = ["easy", "hard"]
    splits = ["train", "validation", "test"]
    for subset in subsets:
        for split in splits:
            preprocess_dataset(huggingface_dataset, subset, split, size=args.train_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=int, default=10000)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)