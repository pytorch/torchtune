# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Launcher and utilities for torchtune recipes

``tune`` provides functionality for launching torchtune recipes as well as local
recipes. Aside from torchtune recipe utilties it integrates with ``torch.distributed.run``
to support distributed job launching by default. ``tune`` offers everyting that ``torchrun``
does with the following additional functionalities:

1. ``tune <recipe> <recipe_args>`` with no optional ``torchrun`` options launches a single python process

2. ``<recipe>`` and recipe arg ``<config>`` can both be passed in as names instead of paths if they're included in torchtune

3. ``tune <path/to/recipe.py> <recipe_args>`` can be used to launch local recipes

4. ``tune <torchrun_options> <recipe> <recipe_args>`` will launch a torchrun job

.. note:: ``tune`` is a python
          `console script <https://packaging.python.org/en/latest/specifications/entry-points/#use-for-scripts>`_
          to the main module
          `scripts.cli_utils.tune <https://github.com/pytorch/torchtune/blob/main/scripts/cli_utils/tune>`_
          declared in the ``scripts`` configuration in
          `setup.py <https://github.com/pytorch/torchtune/blob/main/setup.py>`_.
          It is equivalent to invoking ``python -m scripts.cli_utils.tune``.
"""
import argparse
import runpy
import sys
from pathlib import Path

import torchtune
from torch.distributed.run import get_args_parser, run
from torchtune import list_recipes
from torchtune._cli import list_scripts
from torchtune.utils._distributed import _valid_distributed_single_node_nnodes


def _update_parser_help(parser):
    parser.description = "Torch Tune Recipe Launcher"
    parser.usage = "tune [options] <recipe> [recipe_args]"
    parser.formatter_class = argparse.RawDescriptionHelpFormatter

    # Update torchrun argparse name for more accurate CLI help
    actions = [a.dest for a in parser._actions]
    # Update training_script help to be recipe
    idx = actions.index("training_script")
    parser._actions[idx].dest = "recipe"
    parser._actions[idx].help = "Name or path to recipe to be launched followed by args"

    # Update training_script_args help to be recipe_args
    idx = actions.index("training_script_args")
    parser._actions[idx].dest = "recipe_args"


def _is_distributed_args(args):
    total = len(sys.argv) - 1  # total args minus "tune"
    script_args = len(args.recipe_args) + 1  # script args + 1 for script name
    return total > script_args


def _validate_distributed_args(args):
    """
    Validates nnodes and nproc_per_node are appropriately set for distributed training
    runs.
    """
    if not hasattr(args, "nnodes"):
        raise RuntimeError("Expect --nnodes to be specified for distributed runs")

    if args.nnodes not in _valid_distributed_single_node_nnodes:
        raise RuntimeError(
            f"Expect --nnodes to be one of {_valid_distributed_single_node_nnodes}"
        )

    if not hasattr(args, "nproc_per_node"):
        raise RuntimeError(
            "Expect --nproc_per_node to be specified for distributed runs"
        )

    # TODO (rohan-varma): Add check that nproc_per_node <= cuda device count. Currently,
    # we don't do this since we test on CPUs for distributed. Will update once multi GPU
    # CI is supported.


def main():
    parser = get_args_parser()
    _update_parser_help(parser)
    args = parser.parse_args()

    distributed_args = _is_distributed_args(args)
    cmd = args.recipe
    if not cmd.endswith(".py"):
        pkg_path = Path(torchtune.__file__).parent.absolute()
        if f"{cmd}.py" in list_recipes():
            recipes_pkg_path = pkg_path.parent / "recipes"
            cmd = recipes_pkg_path / f"{cmd}.py"
            args.recipe = str(cmd)

            # Replace config name with package path if provided
            if "--config" in args.recipe_args:
                cfg_idx = args.recipe_args.index("--config") + 1
                config = args.recipe_args[cfg_idx]
                if not config.endswith(".yaml"):
                    args.recipe_args[cfg_idx] = str(
                        recipes_pkg_path / "configs" / f"{config}.yaml"
                    )
        elif cmd in list_scripts():
            cmd = pkg_path / "_cli" / f"{cmd}.py"
            args.recipe = str(cmd)
            assert not distributed_args, "You can't use distributed args with scripts"
        else:
            parser.error(
                f"Unrecognized command '{cmd}'\nTry 'tune --help' for more information."
            )

    if distributed_args:
        _validate_distributed_args(args)
        args.training_script = str(cmd)  # arg names expected by torchrun
        args.training_script_args = args.recipe_args
        run(args)
    else:
        sys.argv = [str(cmd)] + args.recipe_args
        runpy.run_path(str(cmd), run_name="__main__")


if __name__ == "__main__":
    main()
