from datasets import load_dataset
import json

# Load WikiText dataset from huggingface
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Function to convert the dataset to JSON format
def convert_to_json(dataset, output_file):
    json_data = [{"input": sample["text"]} for sample in dataset]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

# Split the data train and test sets
convert_to_json(dataset["train"], "wikitext_train.json")
convert_to_json(dataset["test"], "wikitext_test.json")
