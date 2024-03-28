# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import textwrap

from pathlib import Path

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
            help="Download a model from the HuggingFace Hub.",
            description="Download a model from the HuggingFace Hub.",
            epilog=textwrap.dedent(
                """\
            examples:
                # Download a model from the HuggingFace Hub with a Hugging Face API token
                $ tune download meta-llama/Llama-2-7b-hf --hf-token <TOKEN> --output-dir /tmp/model
                Succesfully downloaded model repo and wrote to the following locations:
                ./model/config.json
                ./model/README.md
                ./model/consolidated.00.pth
                ...

                # Download an ungated model from the HuggingFace Hub
                $ tune download mistralai/Mistral-7B-Instruct-v0.2
                Succesfully downloaded model repo and wrote to the following locations:
                ./model/config.json
                ./model/README.md
                ./model/model-00003-of-00003.safetensors
                ...

            For a list of all models, visit the HuggingFace Hub https://huggingface.co/models.
            """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
            exit_on_error=True,
        )
        self._add_arguments()
        self._parser.set_defaults(func=self._download_cmd)

    def _add_arguments(self) -> None:
        """Add arguments to the parser."""
        self._parser.add_argument(
            "repo_id",
            type=str,
            help="Name of the repository on HuggingFace Hub.",
        )
        self._parser.add_argument(
            "--output-dir",
            type=Path,
            required=False,
            default="./model",
            help="Directory in which to save the model.",
        )
        self._parser.add_argument(
            "--hf-token",
            type=str,
            required=False,
            default=os.getenv("HF_TOKEN", None),
            help="HuggingFace API token. Needed for gated models like Llama2.",
        )

    def _download_cmd(self, args: argparse.Namespace) -> None:
        """Downloads a model from the HuggingFace Hub."""
        # Download the tokenizer and PyTorch model files
        try:
            true_output_dir = snapshot_download(
                args.repo_id,
                local_dir=args.output_dir,
                resume_download=True,
                token=args.hf_token,
            )
        except GatedRepoError:
            self._parser.error(
                "You need to provide a HuggingFace API token to download gated models."
                "You can find your token by visiting https://huggingface.co/settings/tokens"
            )
        except RepositoryNotFoundError:
            self._parser.error(
                f"Repository '{args.repo_id}' not found on the HuggingFace Hub."
            )
        except Exception as e:
            self._parser.error(e)

        print(
            "Succesfully downloaded model repo and wrote to the following locations:",
            *list(Path(true_output_dir).iterdir()),
            sep="\n",
        )
