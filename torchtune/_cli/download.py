# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import textwrap
import traceback

from http import HTTPStatus
from pathlib import Path
from typing import Literal, Union
from warnings import warn

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from kagglehub import model_download
from kagglehub.auth import set_kaggle_credentials
from kagglehub.exceptions import KaggleApiHTTPError
from kagglehub.handle import parse_model_handle
from torchtune._cli.subcommand import Subcommand


class Download(Subcommand):
    """Holds all the logic for the `tune download` subcommand."""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self._parser = subparsers.add_parser(
            "download",
            prog="tune download",
            usage="tune download <repo-id> [OPTIONS]",
            help="Download a model from the Hugging Face Hub or Kaggle Model Hub.",
            description="Download a model from the Hugging Face Hub or Kaggle Model Hub.",
            epilog=textwrap.dedent(
                """\
            examples:
                # Download a model from the Hugging Face Hub with a Hugging Face API token
                $ tune download meta-llama/Llama-2-7b-hf --hf-token <TOKEN>
                Successfully downloaded model repo and wrote to the following locations:
                /tmp/Llama-2-7b-hf/config.json
                /tmp/Llama-2-7b-hf/README.md
                /tmp/Llama-2-7b-hf/consolidated.00.pth
                ...

                # Download an ungated model from the Hugging Face Hub
                $ tune download mistralai/Mistral-7B-Instruct-v0.2 --output-dir /tmp/model
                Successfully downloaded model repo and wrote to the following locations:
                /tmp/model/config.json
                /tmp/model/README.md
                /tmp/model/model-00001-of-00002.bin
                ...

                # Download a model from the Kaggle Model Hub
                $ tune download metaresearch/llama-3.2/pytorch/1b --source kaggle
                Successfully downloaded model repo and wrote to the following locations:
                /tmp/llama-3.2/pytorch/1b/tokenizer.model
                /tmp/llama-3.2/pytorch/1b/params.json
                /tmp/llama-3.2/pytorch/1b/consolidated.00.pth
                ...

            For a list of all models, visit the Hugging Face Hub
            https://huggingface.co/models or Kaggle Model Hub https://kaggle.com/models.
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
            help="Name of the repository on Hugging Face Hub or model handle on Kaggle Model Hub.",
        )
        self._parser.add_argument(
            "--output-dir",
            type=Path,
            required=False,
            default=None,
            help="Directory in which to save the model. Defaults to `/tmp/<model_name>`.",
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
            help="If provided, files matching any of the patterns are not downloaded. Example: '*.safetensors'. "
            "Only supported for Hugging Face Hub models.",
        )
        self._parser.add_argument(
            "--source",
            type=str,
            required=False,
            default="huggingface",
            choices=["huggingface", "kaggle"],
            help="If provided, downloads model weights from the provided repo_id on the designated source hub.",
        )
        self._parser.add_argument(
            "--kaggle-username",
            type=str,
            required=False,
            help="Kaggle username for authentication. Needed for private models or gated models like Llama2.",
        )
        self._parser.add_argument(
            "--kaggle-api-key",
            type=str,
            required=False,
            help="Kaggle API key. Needed for private models or gated models like Llama2. You can find your "
            "API key at https://kaggle.com/settings.",
        )

    def _download_cmd(self, args: argparse.Namespace) -> None:
        # Note: we're relying on argparse to validate if the provided args.source is supported
        if args.source == "huggingface":
            return self._download_from_huggingface(args)
        if args.source == "kaggle":
            return self._download_from_kaggle(args)

    def _download_from_huggingface(self, args: argparse.Namespace) -> None:
        """Downloads a model from the Hugging Face Hub."""
        # Download the tokenizer and PyTorch model files

        # Default output_dir is `/tmp/<model_name>`
        output_dir = args.output_dir
        if output_dir is None:
            model_name = args.repo_id.split("/")[-1]
            output_dir = Path("/tmp") / model_name

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
                local_dir=output_dir,
                local_dir_use_symlinks=output_dir_use_symlinks,
                ignore_patterns=args.ignore_patterns,
                token=args.hf_token,
            )
        except GatedRepoError:
            if args.hf_token:
                self._parser.error(
                    "It looks like you are trying to access a gated repository. Please ensure you "
                    "have access to the repository."
                )
            else:
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
            tb = traceback.format_exc()
            msg = f"Failed to download {args.repo_id} with error: '{e}' and traceback: {tb}"
            self._parser.error(msg)

        print(
            "Successfully downloaded model repo and wrote to the following locations:",
            *list(Path(true_output_dir).iterdir()),
            sep="\n",
        )

    def _download_from_kaggle(self, args: argparse.Namespace) -> None:
        """Downloads a model from the Kaggle Model Hub."""

        # Note: Kaggle doesn't actually use the "repository" terminology, but we still reuse args.repo_id here for simplicity
        model_handle = args.repo_id
        self._validate_kaggle_model_handle(model_handle)
        self._set_kaggle_credentials(args)

        # kagglehub doesn't currently support `local_dir` and `ignore_patterns` like huggingface_hub
        if args.output_dir:
            warn(
                "--output-dir flag is not supported for Kaggle model downloads. "
                "This argument will be ignored."
            )
        if args.ignore_patterns:
            warn(
                "--ignore-patterns flag is not supported for Kaggle model downloads. "
                "This argument will be ignored."
            )

        try:
            output_dir = model_download(model_handle)
            print(
                "Successfully downloaded model repo and wrote to the following locations:",
                *list(Path(output_dir).iterdir()),
                sep="\n",
            )
        except KaggleApiHTTPError as e:
            if e.response.status_code in {
                HTTPStatus.UNAUTHORIZED,
                HTTPStatus.FORBIDDEN,
            }:
                self._parser.error(
                    "It looks like you are trying to access a gated model. Please ensure you "
                    "have access to the model and have provided the proper Kaggle credentials "
                    "using the options `--kaggle-username` and `--kaggle-api-key`. You can also "
                    "set these to environment variables as detailed in "
                    "https://github.com/Kaggle/kagglehub/blob/main/README.md#authenticate."
                )
            elif e.response.status_code == HTTPStatus.NOT_FOUND:
                self._parser.error(
                    f"'{model_handle}' not found on the Kaggle Model Hub."
                )
            tb = traceback.format_exc()
            msg = f"Failed to download {model_handle} with error: '{e}' and traceback: {tb}"
            self._parser.error(msg)
        except Exception as e:
            tb = traceback.format_exc()
            msg = f"Failed to download {model_handle} with error: '{e}' and traceback: {tb}"
            self._parser.error(msg)

    def _validate_kaggle_model_handle(self, handle: str) -> None:
        try:
            parsed_handle = parse_model_handle(handle)
            if (
                parsed_handle.framework == "pytorch"
                and parsed_handle.owner != "metaresearch"
            ):
                warn(
                    f"Requested PyTorch model {handle} was not published from Meta, and therefore "
                    "may not be compatible with torchtune."
                )
            if parsed_handle.framework not in {"pytorch", "transformers"}:
                warn(
                    f"Requested model {handle} is neither a PyTorch nor a Transformers model, and "
                    "therefore may not be compatible with torchtune."
                )
        except Exception as e:
            msg = f"Failed to validate {handle} with error {e}."
            self._parser.error(msg)

    def _set_kaggle_credentials(self, args: argparse.Namespace):
        try:
            if args.kaggle_username or args.kaggle_api_key:
                print(
                    "TIP: you can avoid passing in the --kaggle-username and --kaggle-api-key "
                    "arguments by storing them as the environment variables KAGGLE_USERNAME and "
                    "KAGGLE_KEY, respectively. For more details, see "
                    "https://github.com/Kaggle/kagglehub/blob/main/README.md#authenticate"
                )

                # Fallback to known Kaggle environment variables in case user omits one
                #   of the CLI arguments. Note, there's no need to fallback when both
                #   --kaggle-username and --kaggle-api-key are omitted since kagglehub
                #   will check the environment variables itself.
                kaggle_username = (
                    args.kaggle_username
                    if args.kaggle_username
                    else os.environ.get("KAGGLE_USERNAME")
                )
                kaggle_api_key = (
                    args.kaggle_api_key
                    if args.kaggle_api_key
                    else os.environ.get("KAGGLE_KEY")
                )
                set_kaggle_credentials(kaggle_username, kaggle_api_key)
        except Exception as e:
            msg = f"Failed to set Kaggle credentials with error: '{e}'"
            # not all Kaggle downloads require credentials, so there's no need to terminate
            warn(msg)
