
from typing import Optional


def training_steps_completed(
    curr_epoch: int,
    step_within_epoch: int,
    num_steps_per_epoch: int,
    max_steps_per_epoch: Optional[int] = None
) -> int:
    """
    Compute the total number of steps completed thus far. This is useful for
    logging.

    Args:
        curr_epoch (int): Number of epochs completed
        step_within_epoch (int): Number of steps completed within cuttent epoch
        num_steps_per_epoch (int): Number of steps per epoch determined by the dataloader
        max_steps_per_epoch (int): Maximum number of steps per epoch
    """

    # Total number of steps in an epoch is determined by the max_steps_per_epoch
    # param if this is correctly set
    if (
        max_steps_per_epoch is not None and
        max_steps_per_epoch < num_steps_per_epoch
    ):
        num_steps_per_epoch = max_steps_per_epoch

    total_steps = (curr_epoch * num_steps_per_epoch) + step_within_epoch

    return total_steps
