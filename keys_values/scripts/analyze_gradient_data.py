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
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Dict, Set, Iterable, Callable

import numpy as np
import matplotlib.pyplot as plt
import torch


PREFIXES = ("before", "before_fvec", "after", "after_fvec")

FNAME_MASK = "grad_accumulate_iter{iter}_rank{rank}.pth"


def compare_sets(
    a: Set[str],
    b: Set[str],
    name_a: str,
    name_b: str,
    prefix: str = "",
):
    if a != b:
        raise ValueError(
            f"{prefix}Keys of {name_a} and {name_b} are different:\n"
            f"{name_a} / {name_b}: {a.difference(b)}\n"
            f"{name_b} / {name_a}: {b.difference(a)}\n"
        )


@dataclass(frozen=True)
class GradientData:
    before: Dict[str, torch.Tensor]
    before_fvec: Dict[str, torch.Tensor]
    after: Dict[str, torch.Tensor]
    after_fvec: Dict[str, torch.Tensor]

    @staticmethod
    def from_dict(record: Dict[str, torch.Tensor]) -> "GradientData":
        splits = [
            {
                k[(len(prefix) + 1) :]: v
                for k, v in record.items()
                if k.startswith(prefix + ":")
            }
            for prefix in PREFIXES
        ]
        dtypes = {x.dtype for split in splits for x in split.values()}
        assert len(dtypes) == 1, f"Found different dtypes: {dtypes}"
        return GradientData(
            before=splits[0],
            before_fvec=splits[1],
            after=splits[2],
            after_fvec=splits[3],
        )

    def __post_init__(self):
        before_keys = set(self.before.keys())
        for name, keys in (
            ("before_fvec", set(self.before_fvec.keys())),
            ("after", set(self.after.keys())),
            ("after_fvec", set(self.after_fvec.keys())),
        ):
            compare_sets(before_keys, keys, "before", name)

    def dtype(self) -> torch.dtype:
        return next(iter(self.before.values())).dtype

    def keys(self) -> Iterable[str]:
        return self.before.keys()


def test_consistency(records: List[GradientData], prefix: str) -> None:
    prefix += ": "
    dtypes = {record.dtype() for record in records}
    if len(dtypes) != 1:
        raise ValueError(prefix + f"Found different dtypes: {dtypes}")
    keys_rec0 = set(records[0].keys())
    for i, record in enumerate(records[1:]):
        compare_sets(
            keys_rec0,
            set(record.keys()),
            "rank 0",
            f"rank {i}",
            prefix,
        )
    for name in records[0].keys():

        def modify_msg(x: str, rank: int, part: str) -> str:
            return prefix + f"{part}: {name}: rank {rank}: " + x

        values = {
            prefix: [record.__dict__[prefix][name].unsqueeze(0) for record in records]
            for prefix in PREFIXES
        }
        part = "before<->before_fvec"
        print("Testing " + part)
        for rank, (a, b) in enumerate(zip(values["before"], values["before_fvec"])):
            torch.testing.assert_close(
                a,
                b,
                msg=partial(modify_msg, rank=rank, part=part),
            )
        part = "after<->after_fvec"
        print("Testing " + part)
        for rank, (a, b) in enumerate(zip(values["after"], values["after_fvec"])):
            torch.testing.assert_close(
                a,
                b,
                msg=partial(modify_msg, rank=rank, part=part),
            )
        part = "after are equal"
        print("Testing " + part)
        a = values["after"][0]
        for rank, b in enumerate(values["after"]):
            torch.testing.assert_close(
                a,
                b,
                msg=partial(modify_msg, rank=rank, part=part),
            )
        part = "sum(before)<->after"
        print("Testing " + part)
        a = torch.cat(values["before"], dim=0).sum(dim=0, keepdim=True)
        b = values["after"][0]
        # DEBUG
        diffs = torch.abs(a - b).flatten()
        _, ind = torch.topk(diffs, k=32)
        print("Top 32 most different:")
        print(
            torch.cat(
                [x.flatten()[ind].unsqueeze(-1) for x in (a, b)],
                dim=-1,
            )
        )
        over_ranks = torch.cat(
            [x.flatten()[ind].unsqueeze(-1) for x in values["before"]],
            dim=-1,
        )
        print("For these 32 positions: before across ranks:")
        print(over_ranks)
        # END DEBUG
        torch.testing.assert_close(
            a,
            b,
            msg=partial(modify_msg, rank=0, part=part),
        )


def load_data(path: Path, iter: int, num_devices: int) -> List[GradientData]:
    return [
        GradientData.from_dict(
            torch.load(path / FNAME_MASK.format(iter=iter, rank=rank))
        )
        for rank in range(num_devices)
    ]


def plot_histogram_grid(
    data,
    bins="auto",
    figsize=None,
    title=None,
    shared_x=False,
    shared_y=False,
    **hist_kwargs,
) -> None:
    """
    Plot a 2D arrangement of histograms.

    Args:
        data: 3D array where records[i, j, :] contains data for histogram at
            position (i, j)
        bins: Number of bins or binning strategy
        figsize: Figure size (width, height)
        title: Overall figure title
        shared_x: Share x-axis across subplots
        shared_y: Share y-axis across subplots
        **hist_kwargs: Additional arguments passed to ax.hist()

    """
    n_rows, n_cols, _ = data.shape

    if figsize is None:
        figsize = (3 * n_cols, 3 * n_rows)

    # Create figure and subplots
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=shared_x,
        sharey=shared_y,
    )

    # Handle case where there's only one subplot
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Default histogram style
    default_kwargs = {"bins": bins, "edgecolor": "black", "alpha": 0.7}
    default_kwargs.update(hist_kwargs)

    # Plot histogram for each position
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            data = data[i, j, :]

            # Plot histogram
            ax.hist(data, **default_kwargs)
            ax.set_title(f"({i}, {j})")

            # Only show labels on outer edges if sharing axes
            if not shared_x or i == n_rows - 1:
                ax.set_xlabel("Value")
            if not shared_y or j == 0:
                ax.set_ylabel("Frequency")

            ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=16, y=1.00)

    plt.tight_layout()
    plt.show()

    return fig, axes


def plot_histograms(
    records: List[List[GradientData]],
    filter_name: Callable[[str], bool],
    title: str,
):
    num_iters = len(records)
    num_devices = len(records[0])
    names_to_plot = [name for name in records[0][0].keys() if filter_name(name)]
    for name in names_to_plot:
        # 2D plot "before" (iter, rank)
        _title = title + ": Before (iter, rank): " + name
        num_elems = records[0][0].before[name].numel()
        data = np.empty(
            (num_devices, num_iters, num_elems),
            dtype=np.float32,
        )
        for i in range(num_devices):
            for j in range(num_iters):
                data[i, j, :] = records[j][i].before[name].numpy().flatten()
        plot_histogram_grid(data, title=_title, bins=256, color="skyblue")
        # 1D plot "after" (iter)
        _title = title + ": After (iter): " + name
        num_elems = records[0][0].after[name].numel()
        data = np.empty(
            (1, num_iters, num_elems),
            dtype=np.float32,
        )
        for i in range(num_iters):
            data[0, i, :] = records[i][0].after[name].numpy().flatten()
        plot_histogram_grid(data, title=_title, bins=256, color="skyblue")


# Keys: "transformer.h.*.attn.qkv.lora_B"
# - Should be zeros in places for "keys". Remove them?
def main(
    path: Path,
    iters: List[int],
    num_devices: int,
    layers_to_plot: List[int],
) -> None:
    records = []
    for iter in iters:
        print(f"Processing gradient for iteration {iter}")
        records_for_iter = load_data(path, iter, num_devices)
        prefix = f"Iteration {iter}"
        test_consistency(records_for_iter, prefix)
        records.append(records_for_iter)
    for layer in layers_to_plot:
        print(f"Plotting histograms for layer {layer}")
        filter_name = lambda name: name.contains(f"transformer.h.{layer}.")
        plot_histograms(
            records,
            filter_name,
            f"lora_B gradients for layer {layer}",
        )


if __name__ == "__main__":
    path = Path.home() / "tmp" / "debug"
    iters = list(range(8))
    num_devices = 4
    layers_to_plot = [0, 1]
    main(path, iters, num_devices, layers_to_plot)
