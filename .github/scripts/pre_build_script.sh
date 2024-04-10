#!/bin/bash

# Update __version__ in __init__.py using BUILD_VERSION environment variable
python_file="torchtune/__init__.py"
if [[ -n "$BUILD_VERSION" ]]; then
    version_line=$(grep "^__version__ = " "$python_file")
    if [[ -n "$version_line" ]]; then
        echo "$version_line" | sed "s/='.*'$/'='$BUILD_VERSION'/" > tmp && mv tmp "$python_file"
    else
        echo "Error: __version__ line not found in $python_file."
        exit 1
    fi
else
    echo "Error: BUILD_VERSION environment variable is not set or empty."
    exit 1
fi
