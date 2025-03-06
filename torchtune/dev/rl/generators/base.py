from abc import ABC, abstractmethod
from typing import Optional

import torch

from tensordict import TensorClass


class GeneratorInput(TensorClass):
    tokens: torch.Tensor


class GeneratorOutput(TensorClass):
    query_responses: torch.Tensor
    responses: torch.Tensor
    logits: torch.Tensor
    masks: torch.Tensor
    position_ids: torch.Tensor


class GeneratorABC(ABC):

    @abstractmethod
    def setup(self) -> None:
        """Setup method for initializing the generator.

        This method should be called before generation to prepare any necessary
        resources such as key-value caches for transformer models.
        """
        pass

    @abstractmethod
    def generate(self, input: GeneratorInput) -> GeneratorOutput:
        """Generate text autoregressively one token at a time.

        Args:
            input: The GeneratorInput containing tokens to generate from

        Returns:
            GeneratorOutput containing the generated responses, logits, masks, and position IDs
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the generator state including any caches.

        This method should clear any stored states from previous generations
        to ensure fresh generation for new prompts.
        """
        pass

    @abstractmethod
    def update_weights(self, weights) -> None:
        """Update the weights of the underlying model.

        Args:
            weights: The new weights to be applied to the model.

        Returns:
            None
        """
        pass
