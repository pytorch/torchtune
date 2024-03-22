#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script will handle caching of all artifacts needed for a given test run.
# ./cache_assets.sh alone is a no-op
# ./cache_assets --run-recipe-tests will fetch tokenizer and small model checkpoints.
# ./cache_assets --run-regression-tests will fetch tokenizer and 7B model checkpoint.
# ./cache_assets --run-recipe-tests --run-regression-tests will fetch all of the above.
# In all cases, if the files already exist locally they will not be downloaded from S3.

SMALL_MODEL_URLS=(
    "s3://pytorch-multimodal/small-ckpt-01242024"
    "s3://pytorch-multimodal/small-ckpt-tune-03082024.pt"
    "s3://pytorch-multimodal/small-ckpt-meta-03082024.pt"
    "s3://pytorch-multimodal/small-ckpt-hf-03082024.pt"
)
FULL_MODEL_URL=("s3://pytorch-multimodal/llama2-7b-torchtune.pt")
TOKENIZER_URL=("s3://pytorch-multimodal/llama2-7b/tokenizer.model")

LOCAL_DIR="/tmp/test-artifacts"
S3_URLS=()

# Iterate over command-line args
while [[ $# -gt 0 ]]; do
    arg="$1"
    case $arg in
        "--run-recipe-tests")
            # Add URLs for small models
            for url in "${SMALL_MODEL_URLS[@]}"; do
                S3_URLS+=( "$url" )
            done
            shift # Next argument
            ;;
        "--run-regression-tests")
            # Add URL for large model
            S3_URLS+=(
                $FULL_MODEL_URL
            )
            shift # Next argument
            ;;
    esac
done

# If either recipe or regression tests are running,
# fetch the tokenizer
if ! [ -z "$S3_URLS" ]; then
    S3_URLS+=(
        $TOKENIZER_URL
    )
fi

# Sanity check debug log
echo "Expected artifacts for test run are:"
for url in "${S3_URLS[@]}"; do
    echo "$(basename "$url")"
done

# Download relevant files from S3 to local
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
