# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
from pathlib import Path

from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent.resolve()


def read_requirements(file):
    with open(file) as f:
        reqs = f.read()

    return reqs.strip().split("\n")


def _get_version():
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        sha = subprocess.check_output(cmd, cwd=str(ROOT_DIR)).decode("ascii").strip()
    except Exception:
        sha = None

    if "BUILD_VERSION" in os.environ:
        version = os.environ["BUILD_VERSION"]
    else:
        with open(os.path.join(ROOT_DIR, "version.txt"), "r") as f:
            version = f.readline().strip()
        if sha is not None:
            version += "+" + sha[:7]

    if sha is None:
        sha = "Unknown"
    return version, sha


def _export_version(version, sha):
    version_path = ROOT_DIR / "torchtune" / "version.py"
    with open(version_path, "w") as fileobj:
        fileobj.write("__version__ = '{}'\n".format(version))
        fileobj.write("git_version = {}\n".format(repr(sha)))


VERSION, SHA = _get_version()
_export_version(VERSION, SHA)

print("-- Building version " + VERSION)


setup(
    name="torchtune",
    version=VERSION,
    packages=find_packages(exclude=["tests", "tests.*", "recipes"]),
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
