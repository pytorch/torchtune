from torchtune.datasets import TextCompletionDataset


def custom_dataset(tokenizer) -> TextCompletionDataset:
    filter_fn = lambda x: len(x["text"]) > 0
    return TextCompletionDataset(
        tokenizer=tokenizer,
        source="text",
        data_files="/data/users/rafiayub/data/my_data.txt",
        column="text",
        filter_fn=filter_fn,
        split="train",
    )
