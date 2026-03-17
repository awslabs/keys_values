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
from pathlib import Path
from typing import Tuple, Optional, List, Callable

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# User: I'd need Python code to plot a panel of matrix heatmaps. Data comes from a dictionary of PyTorch tensors.
# Columns of the panel correspond to dictionary entries, use the keys as column headers.
# The tensor for a column of heatmaps has shape (num_rows, x, y), there should be num_rows heatmaps per column,
# each coming from a 2D slice. Use separate legends and color palettes per column, but shared over the heatmaps
# in a column. Plot the legends left and right. x and y axis should be the same for heatmaps in a column.
#
# User: Some changes:
#     The y ticks are the same for all panels, also across column (so the final y dimension of the tensors have the same size). Please add tick marks and numbering on the left (so the numbers on the left of the panels in the leftmost column), and number them with the row position, so range(y).
#     The x ticks are the same for all panels in a column (but may be different between columns). Please add tick marks and numbering on the bottom (so the numbers on the bottom of the bottommost panels in each column)
#
# User: Some changes:
#
#     y and x ticks: Don't write all numbers, write 0, 10, 20, ...
#     Plot the legends for each column only on the right of the column, not also on the left. This avoids the legend clashing with the tick numbering
#
# User: Few changes:
#
#     There is quite some whitespace at the top of the plot, can you remove that?
#     Can you leave more space between the columns? Right now, the second column panels are on top of the ticks of the legend of the first column
#
# Feature | Approach |
# |---|---|
# | **Layout** | `GridSpec` with 3 sub-columns per data column: `[left-cbar │ heatmap │ right-cbar]` |
# | **Shared colour scale per column** | `vmin`/`vmax` computed over the whole `(num_rows, x, y)` tensor before plotting |
# | **Single spanning colorbar** | Placeholder axes are hidden; a ghost `host_ax` covering their combined bounding box is used to anchor one `fig.colorbar` call |
# | **Shared axes per column** | `ax.set_xlim` / `ax.set_ylim` set to the same range for every heatmap in a column |
# | **Separate palettes** | One `cmap` string per column, defaulting to a preset rotation |
def plot_heatmap_panel(
    tensor_dict: dict[str, torch.Tensor],
    palettes: list[str] | None = None,
    figsize_per_cell: tuple[float, float] = (3, 3),
    cbar_width_ratio: float = 0.15,
    col_spacing_ratio: float = 0.25,
    title: str | None = None,
    x_tick_spacing: int | Tuple[int, ...] = 1,
    y_tick_spacing: int = 10,
    layer_idxs: Optional[List[int]] = None,
):
    """
    Plot a panel of heatmaps from a dictionary of PyTorch tensors.

    Args:
        tensor_dict:        Dictionary mapping column names to tensors of shape (num_rows, x, y).
                            The y dimension must be the same for all tensors.
        palettes:           List of colormap names, one per column. Defaults to a preset list.
        figsize_per_cell:   (width, height) in inches for each heatmap cell.
        cbar_width_ratio:   Width of colorbar column relative to a heatmap column.
        col_spacing_ratio:  Width of the spacer inserted between data columns, relative
                            to a heatmap column. Increase to add more inter-column space.
        title:              Optional super-title for the figure.
        x_tick_spacing:     Spacing between labeled tick marks (default: every 1).
        y_tick_spacing:     Spacing between labeled tick marks (default: every 10).
    """
    keys = list(tensor_dict.keys())
    num_cols = len(keys)

    # --- Validate and convert tensors -----------------------------------------
    arrays: dict[str, np.ndarray] = {}
    for key, tensor in tensor_dict.items():
        if tensor.ndim != 3:
            raise ValueError(
                f"Tensor '{key}' has shape {tuple(tensor.shape)}, expected (num_rows, x, y)."
            )
        arrays[key] = tensor.detach().cpu().float().numpy()

    num_rows = arrays[keys[0]].shape[0]
    y_size = arrays[keys[0]].shape[2]  # shared across all columns

    for key, arr in arrays.items():
        if arr.shape[0] != num_rows:
            raise ValueError(
                f"All tensors must have the same num_rows. "
                f"'{key}' has {arr.shape[0]} rows, expected {num_rows}."
            )
        if arr.shape[2] != y_size:
            raise ValueError(
                f"All tensors must have the same y dimension. "
                f"'{key}' has y={arr.shape[2]}, expected {y_size}."
            )

    # --- Color palettes -------------------------------------------------------
    default_palettes = ["viridis", "plasma", "cividis", "magma", "inferno", "coolwarm"]
    if palettes is None:
        palettes = [
            default_palettes[i % len(default_palettes)] for i in range(num_cols)
        ]
    elif len(palettes) != num_cols:
        raise ValueError(
            f"Length of 'palettes' ({len(palettes)}) must match number of columns ({num_cols})."
        )

    # Per-column shared colour range and x size
    col_vmins = {key: float(arrays[key].min()) for key in keys}
    col_vmaxs = {key: float(arrays[key].max()) for key in keys}
    col_x_sizes = {key: arrays[key].shape[1] for key in keys}

    # Shared y tick positions: 0, tick_spacing, 2*tick_spacing, ...
    y_tick_positions = np.arange(0, y_size, y_tick_spacing)

    # --- Figure layout --------------------------------------------------------
    # Sub-columns per data column: [heatmap | cbar]
    # Between data columns (except after the last): [spacer]
    # Full pattern for 3 data cols: heatmap cbar spacer heatmap cbar spacer heatmap cbar
    heatmap_w = figsize_per_cell[0]
    cbar_w = heatmap_w * cbar_width_ratio
    spacer_w = heatmap_w * col_spacing_ratio
    cell_h = figsize_per_cell[1]

    width_ratios = []
    for i in range(num_cols):
        width_ratios.append(heatmap_w)
        width_ratios.append(cbar_w)
        if i < num_cols - 1:
            width_ratios.append(spacer_w)

    # Number of GridSpec columns
    n_gs_cols = num_cols * 2 + (num_cols - 1)  # heatmap+cbar pairs + spacers

    fig_width = sum(width_ratios)
    fig_height = num_rows * cell_h

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = gridspec.GridSpec(
        nrows=num_rows,
        ncols=n_gs_cols,
        figure=fig,
        width_ratios=width_ratios,
        hspace=0.05,
        wspace=0.0,  # spacing now handled by explicit spacer columns
    )

    # Map data-column index → GridSpec column indices
    # col 0: gs 0 (heatmap), 1 (cbar)
    # col 1: gs 3 (heatmap), 4 (cbar)   [gs 2 = spacer]
    # col k: gs k*3 (heatmap), k*3+1 (cbar)
    def gs_heatmap_col(col_idx: int) -> int:
        return col_idx * 3

    def gs_cbar_col(col_idx: int) -> int:
        return col_idx * 3 + 1

    # --- Plot -----------------------------------------------------------------
    for col_idx, key in enumerate(keys):
        arr = arrays[key]
        x_size = col_x_sizes[key]
        cmap = palettes[col_idx]
        vmin = col_vmins[key]
        vmax = col_vmaxs[key]
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        is_leftmost_col = col_idx == 0
        if isinstance(x_tick_spacing, tuple):
            spacing = x_tick_spacing[col_idx]
        else:
            spacing = x_tick_spacing
        x_tick_positions = np.arange(0, x_size, spacing)

        for row_idx in range(num_rows):
            ax = fig.add_subplot(gs[row_idx, gs_heatmap_col(col_idx)])

            # Transpose: tensor-x → imshow x-axis, tensor-y → imshow y-axis
            ax.imshow(
                arr[row_idx].T,
                cmap=cmap,
                norm=norm,
                aspect="auto",
                interpolation="nearest",
                origin="upper",
                extent=[-0.5, x_size - 0.5, y_size - 0.5, -0.5],
            )

            is_bottom_row = row_idx == num_rows - 1

            # X-axis: tick marks everywhere, labels only on bottom row
            ax.set_xticks(x_tick_positions)
            if is_bottom_row:
                ax.set_xticklabels(x_tick_positions, fontsize=7)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", bottom=True, top=False)

            # Y-axis: tick marks everywhere, labels only on leftmost column
            ax.set_yticks(y_tick_positions)
            if is_leftmost_col:
                ax.set_yticklabels(y_tick_positions, fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="y", which="both", left=True, right=False)

            # Column header above the first row
            if row_idx == 0:
                ax.set_title(key, fontsize=11, fontweight="bold", pad=6)

            # Row label on leftmost column
            if is_leftmost_col:
                if layer_idxs is not None:
                    ylabel = f"layer {layer_idxs[row_idx]}"
                else:
                    ylabel = f"row {row_idx}"
                ax.set_ylabel(ylabel, fontsize=9, labelpad=10)

        # --- Single colorbar on the right of each column ----------------------
        cbar_axes = [
            fig.add_subplot(gs[r, gs_cbar_col(col_idx)]) for r in range(num_rows)
        ]
        _make_spanning_colorbar(
            fig,
            cbar_axes,
            sm,
            location="right",
        )

    # No vertical space reserved for a title unless one is given
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout(rect=[0, 0, 1, 1])

    return fig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_spanning_colorbar(
    fig: plt.Figure,
    axes: list[plt.Axes],
    sm: ScalarMappable,
    location: str = "right",
    label: Optional[str] = None,
):
    """
    Draw a single colorbar spanning the combined bounding box of *axes*.
    The placeholder axes are hidden; a ghost host axis anchors the colorbar.
    """
    for ax in axes:
        ax.set_visible(False)

    fig.canvas.draw()
    bboxes = [ax.get_position() for ax in axes]
    x0 = min(b.x0 for b in bboxes)
    y0 = min(b.y0 for b in bboxes)
    x1 = max(b.x1 for b in bboxes)
    y1 = max(b.y1 for b in bboxes)

    host_ax = fig.add_axes([x0, y0, x1 - x0, y1 - y0])
    host_ax.set_visible(False)

    cbar = fig.colorbar(
        sm,
        ax=host_ax,
        location=location,
        fraction=1.0,
        pad=0.0,
    )
    if label is not None:
        cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def main(
    path: Path,
    layer_idxs: List[int],
    steps: List[int],
    figs_fname_template: Optional[str] = None,
    name_filter: Optional[Callable[[str], bool]] = None,
    x_tick_spacing: Optional[int | Tuple[int, ...]] = None,
    y_tick_spacing: Optional[int] = None,
    palettes: Optional[List[str]] = None,
):
    """
    Use this to create plots from data recorded by setting
    `store_weights_rules`, `store_grads_rules` in
    :class:`SizeWeightsGradientsLog`.

    Args:
        path: File stored by the logging tool
        layer_idxs: List of layer indices to plot (rows of the plot)
        steps: One plot is done for each step
        figs_fname_template: If given, the plots are stored to files
            `figs_fname_template.format(step)`
        name_filter: If given, this is a filter for dictionary entries

    """
    if x_tick_spacing is None:
        x_tick_spacing = (1, 4)
    if y_tick_spacing is None:
        y_tick_spacing = 10
    full_tensor_dict = torch.load(path, map_location=torch.device("cpu"))
    if name_filter is not None:
        full_tensor_dict = {k: v for k, v in full_tensor_dict.items() if name_filter(k)}
    if not full_tensor_dict:
        raise ValueError(
            f"Dictionary loaded from {path} has no entries matching name_filter"
        )
    for step in steps:
        tensor_dict = {
            k: torch.cat([v[i : (i + 1)] for i in layer_idxs], dim=0)
            for k, v in full_tensor_dict.items()
            if k.endswith(f"_{step}")
        }
        plot_heatmap_panel(
            tensor_dict,
            figsize_per_cell=(1.9, 1.25),
            col_spacing_ratio=0.6,
            x_tick_spacing=x_tick_spacing,
            y_tick_spacing=y_tick_spacing,
            palettes=palettes,
            layer_idxs=layer_idxs,
        )
        if figs_fname_template is not None:
            fname = figs_fname_template.format(step)
            plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    base_path = Path.home() / "tmp" / "debug" / "sizes"
    # base_fname = "stored_weights"
    base_fname = "stored_gradients"
    path = base_path / (base_fname + ".pth")
    layer_idxs = [0, 1, 2, 3, 8]
    palettes = ["cividis", "coolwarm"]
    steps = list(range(20))
    figs_fname_template = "./" + base_fname + "_{}.png"
    # name_filter = lambda name: "attn_v_weights" in name
    name_filter = lambda name: "bias" in name
    main(
        path,
        layer_idxs,
        steps,
        figs_fname_template,
        name_filter=name_filter,
        x_tick_spacing=(1, 4),
        y_tick_spacing=10,
        palettes=palettes,
    )
