import contextlib
from typing import Callable, Generator, List, Optional, Set, TypeVar

import torch


T = TypeVar("T")


def get_train_context(
    enable_loss_parallel: bool, enable_compiled_autograd: bool
) -> Callable[[Optional[Generator[None, None, None]]], Generator[None, None, None]]:
    """
    Creates a training context that enables loss parallel and compiled autograd as specified.

    Args:
        enable_loss_parallel: Whether to enable loss parallel.
        enable_compiled_autograd: Whether to enable compiled autograd.

    Returns:
        A context manager function that takes an optional context parallel context.
    """

    @contextlib.contextmanager
    def context(cp_context: Generator[None, None, None] | None = None):
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(
                    torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                )

            if cp_context is not None:
                from torch.nn.attention import sdpa_kernel, SDPBackend

                stack.enter_context(
                    sdpa_kernel(
                        [
                            SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION,
                            SDPBackend.CUDNN_ATTENTION,
                        ]
                    )
                )
                stack.enter_context(cp_context)

            yield

    return context


def create_context_parallel_ctx(
    cp_mesh: torch.distributed.DeviceMesh,
    cp_buffers: List[torch.Tensor],
    cp_seq_dims: List[int],
    cp_no_restore_buffers: Set[torch.Tensor],
    cp_rotate_method: str,
) -> Generator[None, None, None]:
    """
    Creates a context parallel context.

    Args:
        cp_mesh: Device mesh for context parallel.
        cp_buffers: List of tensors for context parallel buffers.
        cp_seq_dims: List of sequence dimensions for context parallel buffers.
        cp_no_restore_buffers: Set of tensors that should not be restored.
        cp_rotate_method: Method for rotating in context parallel.

    Returns:
        A context manager for context parallel.
    """
    try:
        from torch.distributed.tensor.experimental import context_parallel
        from torch.distributed.tensor.experimental._attention import set_rotate_method
    except ImportError:
        print(
            f"PyTorch version {torch.__version__} does not include the experimental "
            "Context Parallel API. Please update to a newer version."
        )
        return contextlib.nullcontext()

    set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


def create_consolidated_train_context(
    # Parameters for train context
    enable_loss_parallel: bool,
    enable_compiled_autograd: bool,
    # Parameters for context parallel
    cp_enabled: bool = False,
    cp_mesh: Optional[torch.distributed.DeviceMesh] = None,
    cp_buffers: Optional[List[torch.Tensor]] = None,
    cp_seq_dims: Optional[List[int]] = None,
    cp_no_restore_buffers: Optional[Set[torch.Tensor]] = None,
    cp_rotate_method: Optional[str] = None,
) -> Generator[None, None, None]:
    """
    Creates a consolidated training context that combines loss parallel, compiled autograd,
    and context parallel settings in a single function.

    This function combines the functionality of get_train_context and create_context_parallel_ctx
    into a single context manager.

    Args:
        enable_loss_parallel: Whether to enable loss parallel.
        enable_compiled_autograd: Whether to enable compiled autograd.
        cp_enabled: Whether context parallel is enabled.
        cp_mesh: Device mesh for context parallel.
        cp_buffers: List of tensors for context parallel buffers.
        cp_seq_dims: List of sequence dimensions for context parallel buffers.
        cp_no_restore_buffers: Set of tensors that should not be restored.
        cp_rotate_method: Method for rotating in context parallel.

    Returns:
        A context manager that applies all the specified contexts.

    Example:
        ```python
        with create_consolidated_train_context(
            enable_loss_parallel=parallel_dims.loss_parallel_enabled,
            enable_compiled_autograd=parallelism_config.enable_compiled_autograd,
            cp_enabled=parallel_dims.cp_enabled,
            cp_mesh=world_mesh["cp"] if parallel_dims.cp_enabled else None,
            cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts] if parallel_dims.cp_enabled else None,
            cp_seq_dims=[1, 1] + [0 for _ in model_parts] if parallel_dims.cp_enabled else None,
            cp_no_restore_buffers={inputs, labels} if parallel_dims.cp_enabled else None,
            cp_rotate_method=job_config.parallelism.context_parallel_rotate_method if parallel_dims.cp_enabled else None,
        ):
            # Training code here
        ```
    """

    @contextlib.contextmanager
    def context():
        # Create context parallel context if enabled
        cp_context = None
        if (
            cp_enabled
            and cp_mesh is not None
            and cp_buffers is not None
            and cp_seq_dims is not None
            and cp_rotate_method is not None
        ):
            cp_context = create_context_parallel_ctx(
                cp_mesh=cp_mesh,
                cp_buffers=cp_buffers,
                cp_seq_dims=cp_seq_dims,
                cp_no_restore_buffers=cp_no_restore_buffers or set(),
                cp_rotate_method=cp_rotate_method,
            )

        # Create and enter the train context with the optional cp_context
        train_context = get_train_context(
            enable_loss_parallel=enable_loss_parallel,
            enable_compiled_autograd=enable_compiled_autograd,
        )

        with train_context(cp_context):
            yield

    return context()
