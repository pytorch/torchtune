
# torchtune

**Note: This repository is currently under heavy development.**

The torchtune package contains tools and utilities to finetune generative models with native PyTorch techniques.

# Unit Tests Status

[![Unit Test](https://github.com/pytorch-labs/torchtune/actions/workflows/unit_test.yaml/badge.svg?branch=main)](https://github.com/pytorch-labs/torchtune/actions/workflows/unit_test.yaml)

# Recipe Integration Test Status

![Recipe Integration Test](https://github.com/pytorch-labs/torchtune/actions/workflows/recipe_integration_test.yaml/badge.svg)

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
from torchtune.modules import TransformerDecoder
```

# Quickstart

### Running recipes

On a single GPU
```
tune finetune_llm --config alpaca_llama2_finetune
```

On multiple GPUs using FSDP
```
tune --nnodes 1 --nproc_per_node 4 finetune_llm --config alpaca_llama2_finetune --fsdp True
```

### Copy and edit a custom recipe

To copy a recipe to customize it yourself and then run
```
tune recipe cp finetune_llm my_recipe/finetune_llm.py
tune config cp alpaca_llama2_finetune my_recipe/alpaca_llama2_finetune.yaml
tune my_recipe/finetune_llm.py --config my_recipe/alpaca_llama2_finetune.yaml
```

### Command Utilities

``tune`` provides functionality for launching torchtune recipes as well as local
recipes. Aside from torchtune recipe utilties, it integrates with ``torch.distributed.run``
to support distributed job launching by default. ``tune`` offers everyting that ``torchrun``
does with the following additional functionalities:

1. ``tune <recipe> <recipe_args>`` with no optional ``torchrun`` options launches a single python process

2. ``<recipe>`` and recipe arg ``<config>`` can both be passed in as names instead of paths if they're included in torchtune

3. ``tune <path/to/recipe.py> <recipe_args>`` can be used to launch local recipes

4. ``tune <torchrun_options> <recipe> <recipe_args>`` will launch a torchrun job

5. ``tune recipe`` and ``tune config`` commands provide utilities for listing and copying packaged recipes and configs

# Contributing
### Dev install
To install additional dependencies for contributing, run the following:

```
pip install -e ".[dev]"
```

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

#### Iterate and Serve docs locally

If you're developing locally, you can just open the generated `index.html` file in your browser.

If instead you're using a remote machine, you can use a combination of a simple python HTTP server and port forwarding to serve the docs locally.
This allows you to iterate on the documentation much more quickly than relying on PR
previews.

To do so, after following the above doc build steps, run the following from the `docs/build/html` folder:

```
python -m http.server 8000 # or any free port
```

This will open up a simple HTTP server serving the files in the build directory.
If this is done on a remote machine, you can set up port forwarding from your local machine
to access the server, for example:

```
ssh -L 9000:localhost:8000 $REMOTE_DEV_HOST
```

Now, you can navigate to `localhost:9000` on your local machine to view the rendered documentation.
