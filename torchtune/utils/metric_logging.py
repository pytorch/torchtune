from typing import Mapping, Union

from numpy import ndarray
from torch import Tensor
from typing_extensions import Protocol

Scalar = Union[Tensor, ndarray, int, float]


class MetricLogger(Protocol):
    """Abstract metric logger."""

    def log(
        self,
        name: str,
        data: Scalar,
        step: int,
    ) -> None:
        """Log scalar data.

        Args:
            name (string): tag name used to group scalars
            data (float/int/Tensor): scalar data to log
            step (int): step value to record
        """
        pass

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        """Log multiple scalar values.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int): step value to record
        """
        pass

    def close(self) -> None:
        """
        Close log resource, flushing if necessary.
        Logs should not be written after `close` is called.
        """
        pass


class WandBLogger(MetricLogger):
    """WandB logger."""

    def __init__(self, project: str, entity: str = None, group: str = None, **kwargs):
        """
        Args:
            project (str): WandB project name
            entity (str): WandB entity name
            group (str): WandB group name
        """
        import wandb

        self._wandb = wandb
        self._wandb.init(
            project=project,
            entity=entity,
            group=group,
            reinit=True,
            resume="allow",
            config=kwargs
        )

    def log(self, name: str, data: Scalar, step: int) -> None:
        self._wandb.log({name: data}, step=step)

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        self._wandb.log(payload, step=step)

    def close(self) -> None:
        self._wandb.finish()


class TensorBoardLogger(MetricLogger):
    """
    TensorBoard logger.
    """

    def __init__(self, log_dir: str):
        """
        Args:
            log_dir (str): TensorBoard log directory
        """
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=log_dir)

    def log(self, name: str, data: Scalar, step: int) -> None:
        self._writer.add_scalar(name, data, step)

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        for name, data in payload.items():
            self._writer.add_scalar(name, data, step)

    def close(self) -> None:
        self._writer.close()
