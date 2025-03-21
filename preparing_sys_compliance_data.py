from datasets import load_dataset
import re


dataset = load_dataset('json', data_files='../system-prompt-compliance/output/full_outputs.jsonl', split='train')

# creating a column name new_label for the dataset
def create_new_label(row):
    # Try to find policies in different formats
    # First try numbered format (1., 2., etc)
    policies = re.findall(r'\d+\.\s\*\*[^:]+:\*\*\s+([^\n]+)', row['system_prompt'])
    
    # If no numbered policies found, try bullet point format
    if not policies:
        # Match text after "**" and ":" up to the end of line
        policies = re.findall(r'\*\s\*\*[^:]+:\*\*\s+([^\n]+)', row['system_prompt'])

    n = len(policies)
    result = ["pass"] * n
    explanation = [""] * n

        # Find the index of the target policy
    for i, policy in enumerate(policies):
        if policy.strip() == row['target_policy'].strip():
            if row["label"] != "compliant":
                result[i] = "fail"
                explanation[i] = row["explanation"]
            break

    return {"training_label": result, "training_explanation": explanation}

dataset = dataset.map(create_new_label)
