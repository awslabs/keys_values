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

import pandas as pd

EVAL_METRICS_ALL_FILENAME = "eval_metrics_all.csv"


def _short_task(task: str) -> str:
    return "fin" if task == "final" else task[-3:]


def _sort_entries(entries):
    non_fin = sorted(
        [(st, v) for st, v in entries if st != "fin"], key=lambda x: int(x[0])
    )
    return non_fin + [(st, v) for st, v in entries if st == "fin"]


# We ran evaluations for more than the task for which evaluation loss was
# lowest. With this predicate, we filter for the winning tasks only.
def _filter_dataset_case(
    dataset: str,
    case: str,
    task: str,
) -> bool:
    if dataset.endswith("_128k"):
        # Not yet implemented!!
        return True
    # Filter out error in results:
    if task == "380" and case.startswith("lr_") and dataset.startswith("helmet_trivia"):
        return False
    if task == "fin":
        # Only those for which "fin" is the only result
        return dataset.startswith("helmet_pop") and (
            case.startswith("slr_") or case.startswith("h2onorm_")
        )
    return task != "010"


def main(
    datasets,
    cases,
    result_path,
    final_table: bool,
    multiple_tasks: bool,
):
    if not multiple_tasks and not final_table:
        raise ValueError("If multiple_tasks=False, then final_table must be True")
    base_path = result_path.parent
    col_labels = [
        d.removeprefix("helmet_").rsplit("_", 1)[0].replace("_", r"\_")
        for d in datasets
    ]
    case_labels = [x[1].replace("_", r"\_") for x in cases]

    # table[i][j] = sorted list of (short_task, avg_value) tuples (empty if no file)
    table = []
    for case_key, _ in cases:
        row = []
        for dataset in datasets:
            csv_path = base_path / dataset / case_key / EVAL_METRICS_ALL_FILENAME
            if not csv_path.exists():
                row.append([])
            else:
                df = pd.read_csv(csv_path)
                if multiple_tasks:
                    avg = df.groupby("task")["sub_exact_match"].mean()
                    row.append(
                        _sort_entries(
                            [
                                (_short_task(t), v)
                                for t, v in avg.items()
                                if not final_table
                                or _filter_dataset_case(
                                    dataset, case_key, _short_task(t)
                                )
                            ]
                        )
                    )
                else:
                    avg = df["sub_exact_match"].mean()
                    row.append([(None, avg.item())])
        table.append(row)

    # - final_table == False:
    #   Each dataset gets 2 sub-columns (l for task, r for value) for cross-cell alignment.
    # - final_table == True:
    #   Each dataset column features a single entry (r for value)
    N = len(datasets)
    if final_table:
        col_spec = "l" + "r" * N
        tex_lines = [
            r"\begin{tabular}{" + col_spec + "}",
            r"\noalign{\smallskip}\hline\noalign{\smallskip}",
            " & ".join([""] + col_labels) + r" \\",
            r"\noalign{\smallskip}\hline\hline\noalign{\smallskip}",
        ]
    else:
        col_spec = "l" + "lr" * N
        tex_lines = [
            r"\begin{tabular}{" + col_spec + "}",
            r"\noalign{\smallskip}\hline\noalign{\smallskip}",
            " & ".join(
                [""] + [r"\multicolumn{2}{c}{" + lbl + "}" for lbl in col_labels]
            )
            + r" \\",
            r"\noalign{\smallskip}\hline\hline\noalign{\smallskip}",
        ]
    for case_label, row_entries in zip(case_labels, table):
        max_rows = max((len(e) for e in row_entries), default=0)
        max_rows = max(max_rows, 1)
        if final_table and max_rows > 1:
            print(
                f"{case_label}: max_rows = {max_rows} > 1, must not happen for final_table=True"
            )
        for k in range(max_rows):
            if k == 0 and max_rows > 1:
                label_cell = r"\multirow{" + str(max_rows) + r"}{*}{" + case_label + "}"
            elif k == 0:
                label_cell = case_label
            else:
                label_cell = ""
            cells = [label_cell]
            for entries in row_entries:
                if k < len(entries):
                    st, v = entries[k]
                    if not final_table:
                        cells.append(r"{\small " + st + r":}")
                    cells.append(r"{\small\!" + f"{v * 100:.2f}" + "}")
                else:
                    if not final_table:
                        cells.append("")
                    cells.append("")
            tex_lines.append(" & ".join(cells) + r" \\")
        tex_lines.append(r"\noalign{\smallskip}\hline\noalign{\smallskip}")
    tex_lines.append(r"\end{tabular}")

    if result_path.exists():
        result_path.unlink()
    result_path.write_text("\n".join(tex_lines) + "\n")


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"

    dataset_size = "64k"
    # dataset_size = "128k"
    is_baseline = False
    # is_baseline = True
    if is_baseline:
        base_path = base_path / "baseline"
    datasets = [
        f"helmet_nq_{dataset_size}",
        f"helmet_trivia_qa_{dataset_size}",
        f"helmet_hotpot_qa_{dataset_size}",
        f"helmet_pop_qa_{dataset_size}",
    ]
    if not is_baseline:
        cases = [
            ("lr_4gpu_cs2048_lr5", "lr_2048"),
            ("slr_4gpu_cs2048_lr5", "slr_2048"),
            ("h2o_4gpu_cs2048_lr5", "h2o_2048"),
            ("h2onorm_4gpu_cs2048_lr5", "h2onorm_2048"),
            ("qh2o_4gpu_cs2048_lr5", "qh2o_2048"),
            ("qh2onorm_4gpu_cs2048_lr5", "qh2onorm_2048"),
            ("lr_4gpu_cs1024_lr5", "lr_1024"),
            ("slr_4gpu_cs1024_lr5", "slr_1024"),
            ("h2o_4gpu_cs1024_lr5", "h2o_1024"),
            ("h2onorm_4gpu_cs1024_lr5", "h2onorm_1024"),
        ]
    else:
        cases = [
            ("slr_4gpu_cs1024_lr5", "slr_1024"),
            ("h2onorm_4gpu_cs1024_lr5", "h2onorm_1024"),
        ]
    result_path = base_path / f"results_{dataset_size}.tex"
    # final_table = False
    final_table = True

    main(datasets, cases, result_path, final_table, multiple_tasks=not is_baseline)
