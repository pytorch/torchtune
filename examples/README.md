### torchtune Examples 

This directory contains examples and documentation for getting started with using torchtune's LLM utilities.

#### Llama-7b Inference

`inference.py` contains an example of e2e native inference for a Llama-2 7 billion parameter model. To run the example, please follow the following steps:

1. Install the additional requirements within your conda env via `pip install requirements.txt` (run this command in this directory)
2. Obtain a native Llama-2 7b checkpoint stored on your local machine. This can for example be produced via our conversion utilities, see the [README](https://github.com/pytorch-labs/torchtune/blob/main/torchtune/llm/scripts/checkpoint/README.md) for more details.
3. Run the inference example within your torchtune install:  `python -m examples.inference --native-checkpoint-path <your checkpoint path> --tokenizer-path <your tokenizer path>`
