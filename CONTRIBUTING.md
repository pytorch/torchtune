# Contributing to TorchTune
We want to make contributing to this project as easy and transparent as possible.

&nbsp;

## Dev install
In order to contribute to Torchtune, you should first fork
and then clone your forked repository.

```git clone https://github.com/<YOUR_GITHUB_USER>/torchtune.git```

Then navigate into the newly cloned repo and install dependencies needed for development.

```
cd torchtune
pip install -e ".[dev]"
```

&nbsp;

## Unit Tests and Recipe Tests
For running unit tests locally:
```pytest tests```

For running recipe tests locally (requires access to private S3 bucket):
```./recipes/tests/run_test.sh```

&nbsp;

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

&nbsp;

## Coding Style
`torchtune` uses pre-commit hooks to ensure style consistency and prevent common mistakes. Enable it by:

```
pre-commit install
```

After this pre-commit hooks will be run before every commit.

You can also run this manually on every file using:

```
pre-commit run --all-files
```

&nbsp;

## Build docs

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

&nbsp;

#### Iterate and Serve docs locally

If you're developing locally, you can just open the generated `index.html` file in your browser.

If instead you're using a remote machine, you can use a combination of a simple python HTTP server and port forwarding to serve the docs locally. This allows you to iterate on the documentation much more quickly than relying on PR previews.

To do so, after following the above doc build steps, run the following from the `docs/build/html` folder:

```
python -m http.server 8000 # or any free port
```

This will open up a simple HTTP server serving the files in the build directory. If this is done on a remote machine, you can set up port forwarding from your local machine to access the server, for example:

```
ssh -L 9000:localhost:8000 $REMOTE_DEV_HOST
```

Now, you can navigate to `localhost:9000` on your local machine to view the rendered documentation.

&nbsp;

## Issues
We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe disclosure of security bugs. In those cases, please go through the process outlined on that page and do not file a public issue.

&nbsp;

## License
By contributing to TorchTune, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.

&nbsp;

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

&nbsp;
