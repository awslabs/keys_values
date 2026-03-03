# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file exc ept in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

COLNAME_Q = "q"

COLNAME_STEP = "step"

COLNAME_VALUE = "value"

COLNAME_NAME = "name"


# Prompt:
# I need Python code for plotting. Given are two CSV files. The first, "size_weights.csv", has columns  named step, weight_idx, q,  value. The second, "weight_names.csv", has columns weight_idx, name. Denote the functipon given by the first table by value[step, weight_idx, q]. The code should create plots of a number of curves step -> value[step, weight_idx, q], so step as x axis, value as y axis. There should be several panels for different values of q, which I can set by qs_to_plot. Each panel contains curves for different values of weight_idx, using different colors. The set of weight_idx values represented in the panels should be of size 10, and should be the argmax over weight_idx of max value[:, weight_idx, q_max], where q_max is a fixed input. There should be a legend with names name[weight_idx] from the second table. Note that the 10 weight_idx values for which curves are to be plotted, are the same for all panels.
def main(
    size_table_path: Path,
    names_table_path: Path,
    qs_to_plot: List[str],
    q_max: str,
    top_n: int,
    step_limit: Optional[int] = None,
    colname_idx: str = "weight_idx",
):
    """
    What the code does, step by step

    | Step | Description |
    |------|-------------|
    | **Load** | Reads both CSVs into DataFrames. |
    | **Select top weights** | Filters to `q == q_max`, takes the per-`weight_idx` maximum over all steps, then picks the 10 largest — these indices are used in **every** panel. |
    | **Name lookup** | Builds a `{weight_idx: name}` dict from `weight_names.csv`. |
    | **Colour map** | Assigns one `tab10` colour to each of the 10 selected indices; the same colour appears in all panels. |
    | **Panels** | One subplot per value in `qs_to_plot`, sharing the x-axis. Each panel draws one line per selected `weight_idx`. |
    | **Legend** | Placed outside the top panel to avoid overlapping the curves; shows human-readable names. |

    Adjust `qs_to_plot`, `q_max`, and `top_n` at the top of the file to change the behaviour without touching the rest of the code.
    """

    # ── load data ────────────────────────────────────────────────────────────────
    df = pd.read_csv(size_table_path, dtype={COLNAME_Q: str})
    names = pd.read_csv(names_table_path)
    if step_limit is not None:
        print(f"Filtering: step < {step_limit}")
        df = df[df[COLNAME_STEP] < step_limit]

    # ── select top-10 weight indices ─────────────────────────────────────────────
    # For q == q_max, compute max over steps for every weight_idx, then take top-n
    df_qmax = df[df[COLNAME_Q] == q_max]
    max_per_weight = df_qmax.groupby(colname_idx)[COLNAME_VALUE].max()
    top_weight_ids = max_per_weight.nlargest(top_n).index.tolist()
    print(f"top_weight_ids: {top_weight_ids}")

    # ── build a lookup: weight_idx -> human-readable name ────────────────────────
    idx_to_name = names.set_index(colname_idx)[COLNAME_NAME].to_dict()
    print(f"top_weights: {[idx_to_name[wid] for wid in top_weight_ids]}")

    # ── colour cycle (one colour per weight_idx, shared across all panels) ───────
    cmap = plt.get_cmap("tab10")
    colors = {wid: cmap(i) for i, wid in enumerate(top_weight_ids)}

    # ── filter to only the rows we need ─────────────────────────────────────────
    df_plot = df[df[colname_idx].isin(top_weight_ids) & df[COLNAME_Q].isin(qs_to_plot)]

    # ── plotting ─────────────────────────────────────────────────────────────────
    n_panels = len(qs_to_plot)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(10, 3 * n_panels),
        sharex=True,
        squeeze=False,
    )

    for ax, q_val in zip(axes[:, 0], qs_to_plot):
        df_q = df_plot[df_plot[COLNAME_Q] == q_val]

        for wid in top_weight_ids:
            df_w = df_q[df_q[colname_idx] == wid].sort_values(COLNAME_STEP)
            if df_w.empty:
                continue
            label = idx_to_name.get(wid, f"weight {wid}")
            ax.plot(
                df_w[COLNAME_STEP],
                df_w[COLNAME_VALUE],
                color=colors[wid],
                label=label,
                linewidth=1.5,
            )

        ax.set_title(f"q = {q_val}", fontsize=12)
        ax.set_ylabel(COLNAME_VALUE)
        ax.grid(True, linestyle="--", alpha=0.4)

    # shared x-label on the bottom panel
    axes[-1, 0].set_xlabel(COLNAME_STEP)

    # single legend placed outside the top panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(
        handles,
        labels,
        title="weight",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9,
    )

    plt.suptitle(
        f"Top-{top_n} weights by max value at q={q_max}",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    # plt.savefig("size_weights_plot.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    base_path = Path.home() / "tmp" / "debug" / "sizes"
    # size_table_path = base_path / "size_weights.csv"
    # names_table_path = base_path / "weight_names.csv"
    # colname_idx = "weight_idx"
    size_table_path = base_path / "size_gradients.csv"
    names_table_path = base_path / "gradient_names.csv"
    colname_idx = "grad_idx"
    main(
        size_table_path=size_table_path,
        names_table_path=names_table_path,
        qs_to_plot=["0.010", "0.500", "0.990"],
        q_max="0.990",
        top_n=8,
        step_limit=19,
        colname_idx=colname_idx,
    )
