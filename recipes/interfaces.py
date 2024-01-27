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

    def setup_environment(self, **kwargs) -> None:
        """
        Setup the environment needed for this recipe. This includes
        initializing distributed, setting the right types, devices and seed,
        and initializing the logger.
        """
        ...

    def setup_model(self, **kwargs) -> None:
        """
        Instantiate the model, including additional capabilities such as
        FSDP and activation checkpointing. This does not include loading the
        checkpoint, which should be done through ```load_checkpoint```.
        """
        ...

    def setup_tokenizer(self, **kwargs) -> None:
        """
        Instantiate the tokenizer. This is currently separated out due to our implementation
        of SentencePiece tokenization needed for Llama2.
        """
        ...

    def setup_optimizer_and_loss(self, **kwargs) -> None:
        """
        Instantiate the optimizer (including learning-rate schedulers) and loss functions.
        """
        ...

    def setup_data(self, **kwargs) -> None:
        """
        Set up datasets, samplers and associated dataloaders.
        """
        ...

    def setup_training_params(self, **kwargs) -> None:
        """
        Set up training parameters such as the number of epochs, the number of
        training steps per epoch, the number of steps per logging interval, etc.
        """
        ...

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
