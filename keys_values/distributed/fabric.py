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
import os
from typing import Optional, List, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from keys_values.constants import DEFAULT_MASTER_ADDR, DEFAULT_MASTER_PORT


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
    def spawn(
        func: Callable,
        args: tuple,
        nprocs: int,
    ):
        os.environ.setdefault("MASTER_ADDR", DEFAULT_MASTER_ADDR)
        os.environ.setdefault("MASTER_PORT", DEFAULT_MASTER_PORT)
        mp.spawn(
            func,
            args=args,
            nprocs=nprocs,
            join=True,
        )

    @staticmethod
    def init_process_group_nccl(
        rank: int,
        world_size: int,
    ):
        if Fabric.cuda_is_available():
            torch.cuda.set_device(rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
            )
            if rank != Fabric.rank():
                raise ValueError(f"rank = {rank} != {Fabric.rank()} = Fabric.rank()")
