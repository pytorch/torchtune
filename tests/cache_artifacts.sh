#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script will handle caching of all artifacts needed for a given test run.
# ./cache_artifacts.sh alone is a no-op
# ./cache_artifacts --run-recipe-tests will fetch tokenizer and small model checkpoints.
# ./cache_artifacts --run-regression-tests will fetch tokenizer and 7B model checkpoint.
# ./cache_artifacts --run-recipe-tests --run-regression-tests will fetch all of the above.
# In all cases, if the files already exist locally they will not be downloaded from S3.

SMALL_MODEL_URLS=(
    "https://ossci-datasets.s3.amazonaws.com/torchtune/small-ckpt-tune-03082024.pt"
    "https://ossci-datasets.s3.amazonaws.com/torchtune/small-ckpt-meta-03082024.pt"
    "https://ossci-datasets.s3.amazonaws.com/torchtune/small-ckpt-hf-03082024.pt"
    "https://ossci-datasets.s3.amazonaws.com/torchtune/small-ckpt-tune-llama3-05052024.pt"
    "https://ossci-datasets.s3.amazonaws.com/torchtune/small-ckpt-hf-reward-07122024.pt"
)
FULL_MODEL_URL=("s3://pytorch-multimodal/llama2-7b-torchtune.pt")
TOKENIZER_URLS=(
    "https://ossci-datasets.s3.amazonaws.com/torchtune/tokenizer.model"
    "https://ossci-datasets.s3.amazonaws.com/torchtune/tokenizer_llama3.model"
)

LOCAL_DIR="/tmp/test-artifacts"
S3_URLS=()
S3_OPTS=()

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
        "--silence-s3-logs")
            # Disable S3 progress bar
            S3_OPTS+=(
                "--no-progress"
            )
            shift # Next argument
            ;;
    esac
done

# If either recipe or regression tests are running,
# fetch the tokenizer
if ! [ -z "$S3_URLS" ]; then
    for url in "${TOKENIZER_URLS[@]}"; do
        S3_URLS+=( "$url" )
    done
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
        # S3 files: use s3 cp
        if [[ $S3_URL == s3* ]]; then
            # Download file from S3, optionally silencing progress bar
            cp_cmd="aws s3 cp ${S3_URL} ${LOCAL_DIR}"
            if ! [ -z "$S3_OPTS" ]; then
                cp_cmd="${cp_cmd} ${S3_OPTS}"
            fi
        # For https: download with curl
        else
            cp_cmd="curl --output ${LOCAL_DIR}/${FILE_NAME} ${S3_URL}"
        fi
        bash -c "${cp_cmd}"
        # Check if download was successful
        if [ $? -eq 0 ]; then
            echo "File downloaded successfully: $LOCAL_DIR/$FILE_NAME"
        else
            echo "Failed to download file from S3: $S3_URL"
            exit 1  # Failure exit code
        fi
    fi
done
