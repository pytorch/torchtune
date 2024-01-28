# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Protocol


class FTRecipeInterface(Protocol):
    """
    This class provides a loose structure which every LLM fine-tuning recipe
    should follow. Please note that the interface itself should not be a vehicle for
    code reuse. TorchTune strictly prohibits implementation inheritance in the codebase.

    TODO: Add link to design principle README

    A few notes about the design and the need for this interface:
    - This interface is meant to help recipe-writers organize their code in a way
        which is easy to read, understand and extend.
    - This interface is not meant to add constraints. If the interface comes in the
        way of doing stuff, it needs to be updated or a new interface should be
        written to support what might be a new "family" of recipes.
    """

    def load_checkpoint(self, **kwargs) -> None:
        """
        This method is responsible for loading ALL of the state for the recipe from the
        checkpoint file, including state for the model, optimizer, dataloader and training
        parameters such as the epoch and seed.
        """
        ...

    def save_checkpoint(self, **kwargs) -> None:
        """
        This method is responsible for saving ALL of the state for the recipe,
        including state for the model, optimizer, dataloader and training
        parameters such as the epoch and seed.
        """
        ...

    def train(self, **kwargs) -> None:
        """
        All of the training logic, including the core loop, loss computation, gradient
        accumulation, and backward.
        """
        ...

    def cleanup(self, **kwargs) -> None:
        """
        Any cleaning up needed for the recipe.
        """
        ...
