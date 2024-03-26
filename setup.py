# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from datetime import date

from setuptools import find_packages, setup


def read_requirements(file):
    with open(file) as f:
        reqs = f.read()

    return reqs.strip().split("\n")


def get_nightly_version():
    today = date.today()
    return f"{today.year}.{today.month}.{today.day}"


def parse_args(argv):  # Pass in a list of string from CLI
    parser = argparse.ArgumentParser(description="torchtune setup")
    parser.add_argument(
        "--package_name",
        type=str,
        default="torchtune",
        help="The name of this output wheel",
    )
    return parser.parse_known_args(argv)


if __name__ == "__main__":
    args, unknown = parse_args(sys.argv[1:])

    # Set up package name and version
    name = args.package_name
    is_nightly = "nightly" in name

    version = get_nightly_version() if is_nightly else "0.0.1"

    print(f"-- {name} building version: {version}")

    sys.argv = [sys.argv[0]] + unknown

    setup(
        name="torchtune",
        version=version,
        packages=find_packages(exclude=["tests", "tests.*", "recipes", "recipes.*"]),
        python_requires=">=3.8",
        install_requires=read_requirements("requirements.txt"),
        entry_points={
            "console_scripts": [
                "tune = torchtune._cli.tune:main",
            ]
        },
        description="Package for finetuning LLMs using native PyTorch",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/pytorch/torchtune",
        extras_require={"dev": read_requirements("dev-requirements.txt")},
    )
