
# torchtune

**Note: This repository is currently under heavy development.**

The torchtune package contains tools and utilities to finetune generative models with native PyTorch techniques.

# Installation

Currently, `torchtune` must be built via cloning the repository and installing as follows:

```
git clone https://github.com/pytorch-labs/torchtune
cd torchtune
python setup.py install
```

To verify successful installation, one can run:

```
pip list | grep torchtune
```

And as an example, the following import should work:

```
from torchtune.llm.llama2.transformer import TransformerDecoder
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
