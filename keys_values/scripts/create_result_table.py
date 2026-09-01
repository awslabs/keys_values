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
from typing import Optional

import pandas as pd

from keys_values.evaluation.evaluator import SampleBasedMetricsEvaluator
from keys_values.scripts.cleanup_evaluation import datasets_and_cases

EVAL_METRICS_ALL_FILENAME = "eval_metrics_all.csv"


def _short_task(task: str) -> str:
    return "fin" if task == "final" else task[-3:]


def _sort_entries(entries):
    non_fin = sorted(
        [(st, v) for st, v in entries if st != "fin"], key=lambda x: int(x[0])
    )
    return non_fin + [(st, v) for st, v in entries if st == "fin"]


# Can be used to filter out invalid results
def _filter_dataset_case(
    dataset: str,
    case: str,
    task: str,
) -> bool:
    return True


_PREFIX = "helmet_"

_POSTFIXES = ("_64k", "_128k")


def _metric_name_for_dataset(dataset: str) -> str:
    assert dataset.startswith(_PREFIX)
    len_post = None
    for post in _POSTFIXES:
        if dataset.endswith(post):
            len_post = len(post)
            break
    assert len_post is not None
    key = dataset[len(_PREFIX) : -len_post]
    return SampleBasedMetricsEvaluator.metric_for_helmet_task(key)


def main(
    datasets,
    cases,
    result_path,
    final_table: bool,
    multiple_tasks: bool,
    metric_name: Optional[str] = None,
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
            if metric_name is None:
                _metric_name = _metric_name_for_dataset(dataset)
            else:
                _metric_name = metric_name
            csv_path = base_path / dataset / case_key / EVAL_METRICS_ALL_FILENAME
            if not csv_path.exists():
                row.append([])
            else:
                df = pd.read_csv(csv_path)
                if multiple_tasks:
                    avg = df.groupby("task")[_metric_name].mean()
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
                    avg = df[_metric_name].mean()
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
                    # Metric values of the form 12.3 (one trailing digit)
                    cells.append(r"{\small\!" + f"{v * 100:.1f}" + "}")
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

    metric_name = None  # Select automatically
    # metric_name = "sub_exact_match"  # Override
    # dataset_size = "64k"
    dataset_size = "128k"
    is_rerun = True
    is_baseline = False
    is_base_model = False
    extra_data = False
    filter_dataset = None
    filter_case = None

    if is_rerun:
        base_path = base_path / "rerun"
    elif is_baseline:
        base_path = base_path / "baseline"
    elif is_base_model:
        base_path = base_path / "basemod"
    multiple_tasks = not is_baseline and not is_base_model
    datasets, cases = datasets_and_cases(
        dataset_size,
        extra_data,
        is_baseline,
        is_base_model,
        with_short=True,
        filter_dataset=filter_dataset,
        filter_case=filter_case,
    )
    result_path = base_path / f"results_{dataset_size}.tex"
    # final_table = False
    final_table = True

    main(
        datasets,
        cases,
        result_path,
        final_table,
        multiple_tasks=multiple_tasks,
        metric_name=metric_name,
    )
