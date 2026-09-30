# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import atexit
from functools import partial
import os
import signal
from typing import Optional, List, Callable, Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from lightning.fabric.plugins.environments.lightning import LightningEnvironment
from lightning.fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher


class Fabric:
    """
    Bundles helper functions for distributed processing (training, evaluation).
    Inspired by Lightning Fabric, but we only do basics here.

    Note: We start simple and incomplete, and rather extend this class when
    something else is needed.

    """

    @staticmethod
    def cuda_is_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def is_initialized() -> bool:
        return Fabric.cuda_is_available() and dist.is_initialized()

    @staticmethod
    def device_count() -> int:
        return torch.cuda.device_count() if Fabric.cuda_is_available() else 0

    @staticmethod
    def rank() -> int:
        return dist.get_rank() if Fabric.is_initialized() else 0

    @staticmethod
    def device() -> torch.device:
        return (
            torch.device("cuda", Fabric.rank())
            if Fabric.cuda_is_available()
            else torch.device("cpu")
        )

    @staticmethod
    def world_size() -> int:
        return dist.get_world_size() if Fabric.is_initialized() else 1

    @staticmethod
    def print(msg: str):
        if Fabric.rank() == 0:
            print(msg)

    @staticmethod
    def barrier():
        if Fabric.is_initialized():
            dist.barrier()

    @staticmethod
    def all_reduce_sum(
        x: torch.Tensor,
        group: Optional[List[int]] = None,
    ):
        if Fabric.is_initialized():
            if x.device != Fabric.device():
                raise ValueError(f"x.device = {x.device}, must be {Fabric.device()}")
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)

    @staticmethod
    def all_reduce_mean(
        x: torch.Tensor,
        group: Optional[List[int]] = None,
    ):
        if Fabric.is_initialized():
            if x.device != Fabric.device():
                raise ValueError(f"x.device = {x.device}, must be {Fabric.device()}")
            dist.all_reduce(x, op=dist.ReduceOp.AVG, group=group)

    @staticmethod
    def launch(
        func: Callable,
        nprocs: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Launches processes for distributed training. This is done in the same
        way as for :class:`lightning.fabric.strategies.ddp.DDPStrategy` and its
        default. In particular, we wrap `func` so that a process group is
        initialized via NCCL.

        """
        # Wrapper ensures that process group is initialized (NCCL) at the
        # start of each process.
        wrapped_func = partial(
            wrap_init_process_group,
            func=func,
            world_size=nprocs,
        )
        # These are defaults of Lightning Fabric for DDPStrategy with a single
        # node and no managed cluster.
        cluster_environment = LightningEnvironment()
        launcher = _SubprocessScriptLauncher(
            cluster_environment=cluster_environment,
            num_processes=nprocs,
            num_nodes=1,
        )
        return launcher.launch(wrapped_func, *args, **kwargs)


def wrap_init_process_group(
    func: Callable,
    world_size: int,
    rank: int,
    *args: Any,
    **kwargs: Any,
) -> Any:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    # PyTorch >= 2.4 warns about undestroyed NCCL process group, so we need to do it at program exit
    atexit.register(destroy_process_group)
    return func(rank, *args, **kwargs)


def _distributed_is_initialized() -> bool:
    # `is_initialized` is only defined conditionally
    # https://github.com/pytorch/pytorch/blob/v2.1.0/torch/distributed/__init__.py#L25
    # this might happen to MacOS builds from source (default) or any build from source that sets `USE_DISTRIBUTED=0`
    return dist.is_available() and dist.is_initialized()


def destroy_process_group() -> None:
    # Don't allow Ctrl+C to interrupt this handler
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if _distributed_is_initialized():
        dist.destroy_process_group()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
