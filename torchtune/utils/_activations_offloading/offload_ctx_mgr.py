# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time

from collections import defaultdict

import psutil
import torch
from torch.autograd.graph import saved_tensors_hooks


class ActivationsManager(saved_tensors_hooks):
    """Context manager under which activation tensors created in the forward pass will be managed"""

    def __init__(self, offloading: bool = True, timing: bool = False) -> None:
        self.min_tensor_size_bytes = 1024  # we don't want to bother with small tensors
        self.tracker = (
            {}
        )  # tensor_id = (new_tensor, if_modified)  ---> track what saved/offloaded tensors, are where
        self.tensor_id: int = 0
        self.mem_offload_cache = defaultdict(
            list
        )  # cache of available memory blocks for tensors
        self.gb = 1024 * 1024 * 1024  # bytes in a gigabyte
        self.is_first_forward_call = True
        self.is_first_backward_call = True
        self.is_first_forward_pass = True

        # optional
        self.use_pin_memory: bool = (
            True  # careful with this...we do not yet monitor system memory
        )
        self.virtual_memory_safe_pct = (
            0.60  # we should not exceed this percentage of memory
        )

        # metrics
        self.timing: bool = timing
        self.forward_start_time = 0
        self.backward_start_time = 0
        self.offload_tensors = offloading

        # platform util functions
        def verify_sufficient_virtual_memory():
            curr_pct = get_cpu_ram_pct()
            if curr_pct > self.virtual_memory_safe_pct:
                print(
                    f"***** WARNING: {curr_pct=}% > {self.virtual_memory_safe_pct=}% of virtual memory used"
                )

        def get_cpu_ram_pct() -> float:
            # get the percentage of memory used by the system
            return psutil.virtual_memory().percent

        def get_tensor_id() -> int:
            # create a unique id for each tensor we are managing
            self.tensor_id += 1
            return self.tensor_id

        def get_num_bytes_tensor(x: torch.Tensor) -> int:
            # get the number of bytes in a tensor, for memory management purposes
            return (
                x.element_size() * x.nelement()
            )  # x.element_size() * x._base_storage().nbytes()

        # -------- core pack / unpack work --------
        def pack_tensor(activation: torch.Tensor) -> str:
            # activations are passed in during forward pass - from here we take over and return a unique id
            if self.is_first_forward_call:
                if self.timing:
                    if self.backward_start_time:
                        end_backward_time = time.perf_counter()
                        print(
                            f"***** backward pass took {(end_backward_time - self.backward_start_time):.3f} seconds"
                        )
                    self.forward_start_time = time.perf_counter()

                assert (
                    len(self.tracker) == 0
                ), "backward pass should have cleared tracker of all tensors"

                # set training phase trackers
                self.is_first_forward_call = False
                self.is_first_backward_call = True

            # query for basic tensor info
            activation_dtype = activation.dtype
            num_bytes = get_num_bytes_tensor(activation)
            sizes = activation.size()
            tensor_id = get_tensor_id()

            # skipping small tensors
            if self.offload_tensors and num_bytes >= self.min_tensor_size_bytes:
                cpu_tensor = torch.empty(
                    sizes,
                    dtype=activation_dtype,
                    layout=activation.layout,
                    pin_memory=self.use_pin_memory,
                    device=torch.device("cpu"),
                )

                cpu_tensor.copy_(activation, non_blocking=True)
                self.tracker[tensor_id] = (
                    cpu_tensor,
                    True,
                )  # True = (in future) modified
                return tensor_id
            else:
                print(
                    f"skipping activation of {num_bytes}, size= {sizes}, {activation_dtype=}"
                )
                # gpu_clone = activation.clone().detach()
                self.tracker[tensor_id] = (
                    activation,
                    False,
                )  # True = (in future) modified
                return tensor_id

        def unpack_tensor(unpack_tensor_id: int) -> torch.Tensor:
            # backward pass - we are called with the tensor_id.
            # We then use the tensor_id to retrieve the saved/offloaded tensor
            # and return it in original state (or near original if quantized)
            if self.is_first_backward_call:
                if self.is_first_forward_pass:
                    self.is_first_forward_pass = False
                    if self.use_pin_memory:
                        verify_sufficient_virtual_memory()

                self.is_first_backward_call = False
                self.is_first_forward_call = True
                if self.timing:
                    end_forward_time = time.perf_counter()
                    print(
                        f"***** forward took {(end_forward_time - self.forward_start_time):.3f} seconds"
                    )
                    print(f"***** first backward, managing {len(self.tracker)} tensors")
                    self.backward_start_time = time.perf_counter()

            # retrieve the saved/offloaded tensor
            assert (
                unpack_tensor_id in self.tracker
            ), f"untracked tensor, {unpack_tensor_id}"
            maybe_gpu_tensor, modified = self.tracker[unpack_tensor_id]
            if modified:
                gpu_tensor = maybe_gpu_tensor.to(device="cuda", non_blocking=True)
                del self.tracker[unpack_tensor_id]
                return gpu_tensor
            # clear tensor from tracking
            del self.tracker[unpack_tensor_id]
            return maybe_gpu_tensor

        super().__init__(pack_tensor, unpack_tensor)
