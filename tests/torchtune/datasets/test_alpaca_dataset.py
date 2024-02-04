from unittest.mock import patch

import pytest

from torchtune import datasets
from torchtune.datasets.alpaca import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules.tokenizer import Tokenizer

from tests.test_utils import get_assets_path


class TestAlpacaDataset:
    @pytest.fixture
    def tokenizer(self):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        return Tokenizer.from_file(str(get_assets_path() / "m.model"))

    @patch("torchtune.datasets.alpaca.load_dataset")
    def test_prompt_generation(self, load_dataset, tokenizer):
        """
        Test the the prompt generation based on the alpaca template is correct.
        """

        # mock the call to HF datasets
        load_dataset.return_value =(
            [
                {
                    "instruction": "Give three tips for staying healthy.",
                    "input": "",
                    "output": (
                        f'1.Eat a balanced diet and make sure to include plenty of fruits and vegetables.'
                        f'2. Exercise regularly to keep your body active and strong.'
                        f'3. Get enough sleep and maintain a consistent sleep schedule.'
                    )
                },
                {
                    "instruction": "Evaluate this sentence for spelling and grammar mistakes",
                    "input": "He finnished his meal and left the resturant",
                    "output": "He finished his meal and left the restaurant."
                }
            ]
        )

        # Expected prompts are taken from the "output" field in
        # https://huggingface.co/datasets/tatsu-lab/alpaca
        expected_prompts = [
            (
                f'Below is an instruction that describes a task. Write a response that appropriately '
                f'completes the request.\n\n'
                f'### Instruction:\nGive three tips for staying healthy.\n\n'
                f'### Response:'
            ),
            (
                f'Below is an instruction that describes a task, paired with an input that provides further context. '
                f'Write a response that appropriately completes the request.\n\n'
                f'### Instruction:\nEvaluate this sentence for spelling and grammar mistakes\n\n'
                f'### Input:\nHe finnished his meal and left the resturant\n\n'
                f'### Response:'
            )
        ]

        alpaca_dataset = datasets.get_dataset("alpaca", tokenizer=tokenizer)

        # alpaca_dataset._data contains the raw data loaded from HF's dataset. We need the raw data
        # to test the prompt generation since calling __get__item on the alpaca_dataset object will
        # return the encoded input and label
        for idx, sample in enumerate(alpaca_dataset._data):
            assert (
                expected_prompts[idx] == alpaca_dataset._generate_prompt(sample["instruction"], sample["input"])
            )

    @patch("torchtune.datasets.alpaca.load_dataset")
    def test_label_no_masking(self, load_dataset, tokenizer):
        """
        Test whether the input and the labels are correctly created when the input is not masked.
        """

        # mock the call to HF datasets
        load_dataset.return_value =(
            [
                {
                    "instruction": "Give three tips for staying healthy.",
                    "input": "",
                    "output": (
                        f'1.Eat a balanced diet and make sure to include plenty of fruits and vegetables.'
                        f'2. Exercise regularly to keep your body active and strong.'
                        f'3. Get enough sleep and maintain a consistent sleep schedule.'
                    )
                }
            ]
        )

        alpaca_dataset = datasets.get_dataset("alpaca", tokenizer=tokenizer)
        input, labels = alpaca_dataset[0]

        assert len(input) == len(labels)
        assert labels[-1] == tokenizer.eos_id
        assert input[0] == tokenizer.bos_id
        assert CROSS_ENTROPY_IGNORE_IDX not in labels

    @patch("torchtune.datasets.alpaca.load_dataset")
    def test_label_masking(self, load_dataset, tokenizer):
        """
        Test whether the input and the labels are correctly created when the input is masked.
        """

        # mock the call to HF datasets
        load_dataset.return_value =(
            [
                {
                    "instruction": "Give three tips for staying healthy.",
                    "input": "",
                    "output": (
                        f'1.Eat a balanced diet and make sure to include plenty of fruits and vegetables.'
                        f'2. Exercise regularly to keep your body active and strong.'
                        f'3. Get enough sleep and maintain a consistent sleep schedule.'
                    )
                }
            ]
        )

        alpaca_dataset = datasets.get_dataset("alpaca", tokenizer=tokenizer, train_on_input=False)

        # Extract the prompt and tokenize it; we'll need this to test whether we're masking the
        # input correctly
        sample = alpaca_dataset._data[0]
        prompt = alpaca_dataset._generate_prompt(sample["instruction"], sample["input"])
        encoded_prompt = tokenizer.encode(text=prompt, add_bos=True, add_eos=False)

        # Generate the input and labels
        input, labels = alpaca_dataset[0]

        assert len(input) == len(labels)
        assert labels[-1] == tokenizer.eos_id
        assert input[0] == tokenizer.bos_id
        assert labels.count(CROSS_ENTROPY_IGNORE_IDX) == len(encoded_prompt)

    @patch("torchtune.datasets.alpaca.load_dataset")
    def test_alpaca_clean(self, load_dataset, tokenizer):
        """
        Test whether the input and the labels are correctly created when the input is not masked.
        """

        # mock the call to HF datasets
        load_dataset.return_value =(
            [
                {
                    "instruction": "Give three tips for staying healthy.",
                    "input": "",
                    "output": (
                        f'1.Eat a balanced diet and make sure to include plenty of fruits and vegetables.'
                        f'2. Exercise regularly to keep your body active and strong.'
                        f'3. Get enough sleep and maintain a consistent sleep schedule.'
                    )
                }
            ]
        )

        alpaca_dataset = datasets.get_dataset("alpaca", tokenizer=tokenizer, use_clean=True)
        input, labels = alpaca_dataset[0]

        assert len(input) == len(labels)
        assert labels[-1] == tokenizer.eos_id
        assert input[0] == tokenizer.bos_id
        assert CROSS_ENTROPY_IGNORE_IDX not in labels
