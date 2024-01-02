
# torchtune

**Note: This repository is currently under heavy development.**

The torchtune package contains tools and utilities to finetune generative models with native PyTorch techniques.

# Installation

This library requires PyTorch >= 2.0. Please install locally using [this guide](https://pytorch.org/get-started/locally/).

Currently, `torchtune` must be built via cloning the repository and installing as follows:

```
git clone https://github.com/pytorch-labs/torchtune
cd torchtune
pip install -e .
```

To verify successful installation, one can run:

```
tune recipe list
```

And as an example, the following import should work:

```
from torchtune.models.llama2.transformer import TransformerDecoder
```

# Running recipes

On a single GPU
```
tune finetune_llm --config alpaca_llama2_finetune
```

On multiple GPUs using FSDP
```
tune --nnodes 1 --nproc_per_node 4 finetune_llm --config alpaca_llama2_finetune --fsdp True
```

To copy a recipe to customize it yourself and then run
```
tune recipe cp -r finetune_llm -p my_recipe/finetune_llm.py
tune config cp -c alpaca_llama2_finetune -p my_recipe/alpaca_llama2_finetune.yaml
tune my_recipe/finetune_llm.py --config my_recipe/alpaca_llama2_finetune.yaml
```

# Quickstart
# Contributing
### Coding Style
`torchtune` uses pre-commit hooks to ensure style consistency and prevent common mistakes. Enable it by:

```
pre-commit install
```

After this pre-commit hooks will be run before every commit.

You can also run this manually on every file using:

```
pre-commit run --all-files
```

### Build docs

From the `docs` folder:

Install dependencies:

```
pip install -r requirements.txt
```

Then:

```
make html
# Now open build/html/index.html
```

To avoid building the examples (which execute python code and can take time) you
can use `make html-noplot`. To build a subset of specific examples instead of
all of them, you can use a regex like `EXAMPLES_PATTERN="plot_the_best_example*"
make html`.

If the doc build starts failing for a weird reason, try `make clean`.
