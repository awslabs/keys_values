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
from typing import Optional

import torch


class WeightsTracker:
    """
    Provides wrapper around tensor which checks for NaN and large values.
    If the condition is met, the tensor is stored, and an exception is
    raised.

    """

    def __init__(self):
        self.file_name: Optional[str] = None
        self.threshold: Optional[float] = None

    def initialize(self, file_name: str, threshold: Optional[float] = None):
        self.file_name = file_name
        self.threshold = threshold

    def __call__(self, x: torch.Tensor):
        if self.file_name is not None:
            # Check for NaNs
            x_det = x.detach()
            if torch.isnan(x_det).any():
                torch.save(x_det, self.file_name)
                raise AssertionError(
                    f"Tensor: {torch.isnan(x_det).sum()} of {x_det.numel()} entries are NaN"
                )
            if self.threshold is not None:
                if (torch.abs(x_det) > self.threshold).any():
                    torch.save(x_det, self.file_name)
                    raise AssertionError(
                        f"Tensor: No NaNs, but {(torch.abs(x_det) > self.threshold).sum()} of {x_det.numel()} entries are |elem| > {self.threshold}"
                    )
        return x


WEIGHTS_TRACKER = WeightsTracker()


def initialize_weights_tracker(
    file_name: str,
    threshold: Optional[float] = None,
):
    """
    Initializes global weights tracker.

    Args:
        file_name: Tensor written there if condition is met
        threshold: If given, we don't just check for NaNs, but also for entries
            with absolute value larger than this.

    """
    WEIGHTS_TRACKER.initialize(file_name, threshold)


def track(x: torch.Tensor) -> torch.Tensor:
    """
    Wrap any tensor you like to track.

    Note: Using this tooling needs quite a bit of work (annotating all
    steps in question). Try to use `torch.autograd.detect_anomaly` first,
    wrapping the backward computations in
    `keys_values.kvcache.gradient.accumulate`.

    """
    return WEIGHTS_TRACKER(x)
