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


def main(datasets, cases, result_path):
    base_path = result_path.parent
    col_labels = [d.removeprefix("helmet_") for d in datasets]
    case_labels = [x[1] for x in cases]

    # table[i][j] = list of "short_task:avg" strings (empty list if no file)
    table = []
    for case_key, _ in cases:
        row = []
        for dataset in datasets:
            csv_path = base_path / dataset / case_key / EVAL_METRICS_ALL_FILENAME
            if not csv_path.exists():
                row.append([])
            else:
                df = pd.read_csv(csv_path)
                avg = df.groupby("task")["sub_exact_match"].mean()
                row.append([f"{_short_task(t)}:{v:.4f}" for t, v in avg.items()])
        table.append(row)

    col_spec = "l" + "c" * len(datasets)
    tex_lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\hline",
        " & ".join([""] + col_labels) + r" \\",
        r"\hline",
    ]
    for i, case_label in enumerate(case_labels):
        cells = [case_label]
        for cell_lines in table[i]:
            if not cell_lines:
                cells.append("")
            elif len(cell_lines) == 1:
                cells.append(cell_lines[0])
            else:
                cells.append(r"\makecell{" + r" \\ ".join(cell_lines) + "}")
        tex_lines.append(" & ".join(cells) + r" \\")
    tex_lines += [r"\hline", r"\end{tabular}"]

    result_path.write_text("\n".join(tex_lines) + "\n")


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"

    dataset_size = "64k"
    datasets = [
        f"helmet_nq_{dataset_size}",
        f"helmet_trivia_qa_{dataset_size}",
        f"helmet_hotpot_qa_{dataset_size}",
        f"helmet_pop_qa_{dataset_size}",
    ]
    cases = [
        ("lr_4gpu_cs2048_lr5", "lr_2048"),
        ("h2o_4gpu_cs2048_lr5", "h2o_2048"),
        ("slr_4gpu_cs2048_lr5", "slr_2048"),
        ("qh2o_4gpu_cs2048_lr5", "qh2o_2048"),
        ("h2onorm_4gpu_cs2048_lr5", "h2onorm_2048"),
        ("qh2onorm_4gpu_cs2048_lr5", "qh2onorm_2048"),
        ("lr_4gpu_cs1024_lr5", "lr_1024"),
        ("h2o_4gpu_cs1024_lr5", "h2o_1024"),
    ]
    result_path = base_path / f"results_{dataset_size}.tex"

    main(datasets, cases, result_path)
