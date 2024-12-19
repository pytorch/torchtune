#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

NIGHTLY_VERSION="dev20241121"

# Install pytorch nightly for export-friendly modules to run.
pip install torch==2.6.0.${NIGHTLY_VERSION} torchvision==0.20.0.${NIGHTLY_VERSION} --extra-index-url https://download.pytorch.org/whl/nightly/cpu
