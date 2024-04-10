#!/bin/bash

# Update __version__ in __init__.py using BUILD_VERSION environment variable
python_file="torchtune/__init__.py"
if [[ -n "$BUILD_VERSION" ]]; then
    sed "s/^__version__ = .*/__version__ = '$BUILD_VERSION'/" "$python_file" > tmp && mv tmp "$python_file"
else
    echo "Error: BUILD_VERSION environment variable is not set or empty."
    exit 1
fi
