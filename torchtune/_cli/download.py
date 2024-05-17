# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import textwrap

from pathlib import Path
from typing import Literal, Union

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from torchtune._cli.subcommand import Subcommand


class Download(Subcommand):
    """Holds all the logic for the `tune download` subcommand."""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self._parser = subparsers.add_parser(
            "download",
            prog="tune download",
            usage="tune download <repo-id> [OPTIONS]",
            help="Download a model from the Hugging Face Hub.",
            description="Download a model from the Hugging Face Hub.",
            epilog=textwrap.dedent(
                """\
            examples:
                # Download a model from the Hugging Face Hub with a Hugging Face API token
                $ tune download meta-llama/Llama-2-7b-hf --hf-token <TOKEN> --output-dir /tmp/model
                Successfully downloaded model repo and wrote to the following locations:
                ./model/config.json
                ./model/README.md
                ./model/consolidated.00.pth
                ...

                # Download an ungated model from the Hugging Face Hub
                $ tune download mistralai/Mistral-7B-Instruct-v0.2
                Successfully downloaded model repo and wrote to the following locations:
                ./model/config.json
                ./model/README.md
                ./model/model-00001-of-00002.bin
                ...

            For a list of all models, visit the Hugging Face Hub https://huggingface.co/models.
            """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self._parser.set_defaults(func=self._download_cmd)

    def _add_arguments(self) -> None:
        """Add arguments to the parser."""
        self._parser.add_argument(
            "repo_id",
            type=str,
            help="Name of the repository on Hugging Face Hub.",
        )
        self._parser.add_argument(
            "--output-dir",
            type=Path,
            required=False,
            default="./model",
            help="Directory in which to save the model.",
        )
        self._parser.add_argument(
            "--output-dir-use-symlinks",
            type=str,
            required=False,
            default="auto",
            help=(
                "To be used with `output-dir`. If set to 'auto', the cache directory will be used and the file will be"
                " either duplicated or symlinked to the local directory depending on its size. It set to `True`, a"
                " symlink will be created, no matter the file size. If set to `False`, the file will either be"
                " duplicated from cache (if already exists) or downloaded from the Hub and not cached."
            ),
        )
        self._parser.add_argument(
            "--hf-token",
            type=str,
            required=False,
            default=os.getenv("HF_TOKEN", None),
            help="Hugging Face API token. Needed for gated models like Llama2.",
        )
        self._parser.add_argument(
            "--ignore-patterns",
            type=str,
            required=False,
            default="*.safetensors",
            help="If provided, files matching any of the patterns are not downloaded. Defaults to ignoring "
            "safetensors files to avoid downloading duplicate weights.",
        )

    def _download_cmd(self, args: argparse.Namespace) -> None:
        """Downloads a model from the Hugging Face Hub."""
        # Download the tokenizer and PyTorch model files

        # Raise if local_dir_use_symlinks is invalid
        output_dir_use_symlinks: Union[Literal["auto"], bool]
        use_symlinks_lowercase = args.output_dir_use_symlinks.lower()
        if use_symlinks_lowercase == "true":
            output_dir_use_symlinks = True
        elif use_symlinks_lowercase == "false":
            output_dir_use_symlinks = False
        elif use_symlinks_lowercase == "auto":
            output_dir_use_symlinks = "auto"
        else:
            self._parser.error(
                f"'{args.output_dir_use_symlinks}' is not a valid value for `--output-dir-use-symlinks`. It must be either"
                " 'auto', 'True' or 'False'."
            )

        print(f"Ignoring files matching the following patterns: {args.ignore_patterns}")
        try:
            true_output_dir = snapshot_download(
                args.repo_id,
                local_dir=args.output_dir,
                local_dir_use_symlinks=output_dir_use_symlinks,
                ignore_patterns=args.ignore_patterns,
                token=args.hf_token,
            )
        except GatedRepoError:
            self._parser.error(
                "It looks like you are trying to access a gated repository. Please ensure you "
                "have access to the repository and have provided the proper Hugging Face API token "
                "using the option `--hf-token` or by running `huggingface-cli login`."
                "You can find your token by visiting https://huggingface.co/settings/tokens"
            )
        except RepositoryNotFoundError:
            self._parser.error(
                f"Repository '{args.repo_id}' not found on the Hugging Face Hub."
            )
        except Exception as e:
            self._parser.error(e)

        print(
            "Successfully downloaded model repo and wrote to the following locations:",
            *list(Path(true_output_dir).iterdir()),
            sep="\n",
        )
