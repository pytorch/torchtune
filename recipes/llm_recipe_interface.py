# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Protocol


class LLMRecipeInterface(Protocol):
    """
    This class provides a general structure which each fine-tuning recipe
    should follow.
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
        checkpoint. All checkpointing should be done in ```load_checkpoint```.
        """
        ...

    def setup_tokenizer(self, **kwargs) -> None:
        """
        Instantiate the tokenizer. Unfortunately, given the current implementation
        for SentencePiece, this does include loading the tokenizer checkpoint.
        """
        ...

    def setup_optimizer_and_loss(self, **kwargs) -> None:
        """
        Instantiates the optimizer and loss functions correctly. This includes making sure
        learing-rate schedulers are correctly setup.
        """
        ...

    def setup_data(self, **kwargs) -> None:
        """
        Logic associated with setting up datasets, samplers and associated dataloaders should
        be added here.
        """
        ...

    def load_checkpoint(self, **kwargs) -> None:
        """
        This method is responsible for loading ALL of the state for the recipe from the
        checkpoint, including state for the model, optimizer, dataloader and training
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
