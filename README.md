
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
pip list | grep torchtune
```

And as an example, the following import should work:

```
from torchtune.models.llama2.transformer import TransformerDecoder
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

### Iterate and Serve docs locally

You can use a combination of a simple python HTTP server and port forwarding to
serve the docs locally on your machine if you're using a remote machine for development.
This allows you to iterate on the documentation much more quickly than relying on PR
previews.

After following the above doc build steps, run the following from the `docs/build/html` folder:

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

To iterate on the documentation, after making your changes, simply run `make html` again (you don't need to bring the server down).
This will update the documentation after the page is refreshed.
