# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path

import requests
from tqdm import tqdm

TORCHTUNE_LOCAL_CACHE_FOLDER = Path("~/.cache/torchtune").expanduser()


def download_file(url: str, save_path: Path, chunk_size: int = 8192):
    """
    Download a file with progress bar.

    Args:
        url (str): URL of the file to download
        save_path (Path): Path where the file should be saved
        chunk_size (int): Size of chunks to download at a time

    Raises:
        requests.RequestException: if download fails
    """
    if save_path.parent is not None and not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(save_path, "wb") as f, tqdm(
            desc=save_path.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = f.write(data)
                progress_bar.update(size)

    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        raise
