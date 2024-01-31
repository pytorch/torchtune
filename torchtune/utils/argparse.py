# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import yaml


class TuneArgumentParser(argparse.ArgumentParser):
    """
    TuneArgumentParser is a helpful utility subclass of the argparse ArgumentParser that
    adds a builtin argument "config". The config argument takes a file path to a yaml file
    and will load in argument defaults from the yaml file. The yaml file must only contain
    argument names and their values and nothing more, it does not have to include all of the
    arguments. These values will be treated as defaults and can still be overridden from the
    command line. Everything else works the same as the base ArgumentParser and you should
    consult the docs for more info.

    https://docs.python.org/3/library/argparse.html

    *Note: This class does not support setting "required" arguments.*
    *Note: This class uses "config" as a builtin argument so it is not available to use*
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().add_argument(
            "--config", type=str, help="Path/name of a yaml file with recipe args"
        )

    def parse_known_args(self, *args, **kwargs):
        """This acts the same as the base parse_known_args but will first load in defaults from
        from the config yaml file if it is provided. The command line args will always take
        precident over the values in the config file. All other parsing method, such as parse_args,
        internally call this method so they will inherit this property too. For more info see
        the docs for the base method.

        https://docs.python.org/3/library/argparse.html#the-parse-args-method
        """
        namespace, _ = super().parse_known_args(*args, **kwargs)
        if namespace.config is not None:
            with open(namespace.config, "r") as f:
                config = yaml.safe_load(f)
            assert "config" not in config, "Cannot use 'config' within a config file"
            assert all(
                c in namespace for c in config
            ), "Provided config contains unrecognized arguments"
            self.set_defaults(**config)
        namespace, unknown_args = super().parse_known_args(*args, **kwargs)
        del namespace.config
        return namespace, unknown_args

    def add_argument(self, *args, **kwargs):
        """This calls the base method but throws an error if the required flag is set or the name used is config.
        For more info on the method see the docs for the base method.

        https://docs.python.org/3/library/argparse.html#the-add-argument-method
        """
        assert (
            "--config" not in args
        ), "--config is a protected argument and cannot be used as an argument name"
        assert not kwargs.get("required", False), "Required not supported"
        return super().add_argument(*args, **kwargs)
