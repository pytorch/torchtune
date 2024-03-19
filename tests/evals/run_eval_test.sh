#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Script to run eval tests

LOCAL_DIR="/tmp/test-artifacts"
S3_URLS=(
    "s3://pytorch-multimodal/llama2-7b/tokenizer.model"
    "s3://pytorch-multimodal/small-ckpt-tune-03082024.pt"
)
PYTEST_COMMAND="pytest tests/evals -s"

mkdir -p $LOCAL_DIR
for S3_URL in "${S3_URLS[@]}"; do
    FILE_NAME=$(basename "$S3_URL")

    # Check if file already exists locally
    if [ -e "$LOCAL_DIR/$FILE_NAME" ]; then
        echo "File already exists locally: $LOCAL_DIR/$FILE_NAME"
    else
        # Download file from S3
        aws s3 cp "$S3_URL" "$LOCAL_DIR"

        # Check if download was successful
        if [ $? -eq 0 ]; then
            echo "File downloaded successfully: $LOCAL_DIR/$FILE_NAME"
        else
            echo "Failed to download file from S3: $S3_URL"
            exit 1  # Failure exit code
        fi
    fi
done

$PYTEST_COMMAND

# Check pytest exit status
if [ $? -eq 0 ]; then
    echo "Pytest passed successfully"
    exit 0  # Success exit code
else
    echo "Pytest failed"
    exit 1  # Failure exit code
fi
