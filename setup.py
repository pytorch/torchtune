# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup


def read_requirements(file):
    with open(file) as f:
        reqs = f.read()

    return reqs.strip().split("\n")


setup(
    name="torchtune",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "tune = torchtune._cli.cli_utils.tune:main",
        ]
    },
    description="Package for finetuning LLMs using native PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch-labs/torchtune",
    extras_require={"dev": read_requirements("dev-requirements.txt")},
)
