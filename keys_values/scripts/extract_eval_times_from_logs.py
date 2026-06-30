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
import csv
import re
import statistics
from pathlib import Path


_EVAL_RE = re.compile(
    r"\[rank (\d+) \|[^\]]*\]: Batch (?:(?:step-\d+|final), )?\[(\d+), (\d+)\]:.*eval_time = ([\d.]+) ms"
)


def _wrap(s: str) -> str:
    return "{\\small\\!" + s + "}"


def _task_max(tasks: list) -> str:
    step_tasks = [t for t in tasks if t.startswith("step-")]
    return max(step_tasks) if step_tasks else "final"


def main(log_dir: Path, base_path: str) -> None:
    bp = re.escape(base_path.rstrip("/")) + "/"
    # subdir present (baseline/basemod): base_path/subdir/dataset/policy/eval/...
    store_re_subdir = re.compile(
        bp + r"(baseline|basemod)/(helmet_[^/]+|longbench_[^/]+)/([^/]+)/eval/eval_metrics_\d+\.csv"
    )
    # subdir absent, task present: base_path/dataset/policy/task/eval/...
    store_re_task = re.compile(
        bp + r"(helmet_[^/]+|longbench_[^/]+)/([^/]+)/(step-\d{6}|final)/eval/eval_metrics_\d+\.csv"
    )
    all_rows = []

    for i in range(4):
        log_file = log_dir / f"gpu{i}.log"
        if not log_file.exists():
            continue
        print(f"Loading {log_file}")
        with log_file.open() as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            m_eval = _EVAL_RE.search(line)
            if not m_eval:
                continue
            if idx + 1 >= len(lines):
                continue

            next_line = lines[idx + 1]
            m_subdir = store_re_subdir.search(next_line)
            m_task = store_re_task.search(next_line)
            if not m_subdir and not m_task:
                continue

            rank = int(m_eval.group(1))
            id1 = int(m_eval.group(2))
            id2 = int(m_eval.group(3))
            time_secs = float(m_eval.group(4)) / 1000.0

            if m_subdir:
                all_rows.append(
                    {
                        "dataset": m_subdir.group(2),
                        "policy": m_subdir.group(3),
                        "subdir": m_subdir.group(1),
                        "task": "",
                        "rank": rank,
                        "id1": id1,
                        "id2": id2,
                        "time_secs": time_secs,
                    }
                )
            else:
                all_rows.append(
                    {
                        "dataset": m_task.group(1),
                        "policy": m_task.group(2),
                        "subdir": "",
                        "task": m_task.group(3),
                        "rank": rank,
                        "id1": id1,
                        "id2": id2,
                        "time_secs": time_secs,
                    }
                )

    if not all_rows:
        print("No data found.")
        return

    subdir = all_rows[0]["subdir"]
    has_task = not subdir
    stem = f"times_eval_{subdir}" if subdir else "times_eval"

    if has_task:
        fieldnames = ["dataset", "policy", "task", "rank", "id1", "id2", "time_secs"]
    else:
        fieldnames = ["dataset", "policy", "rank", "subdir", "id1", "id2", "time_secs"]

    csv_path = log_dir / f"{stem}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    datasets = list(dict.fromkeys(r["dataset"] for r in all_rows))
    policies = list(dict.fromkeys(r["policy"] for r in all_rows))

    times_by_combo = {}
    if has_task:
        entries_by_combo = {}
        for r in all_rows:
            entries_by_combo.setdefault((r["dataset"], r["policy"]), []).append(
                (r["task"], r["time_secs"])
            )
        for combo, entries in entries_by_combo.items():
            tmax = _task_max([e[0] for e in entries])
            print(f"{combo}: tmax = {tmax}")
            times = [t for task, t in entries if task == tmax]
            if times:
                times_by_combo[combo] = times
    else:
        for r in all_rows:
            times_by_combo.setdefault((r["dataset"], r["policy"]), []).append(r["time_secs"])

    tex_lines = [
        r"\begin{tabular}{l" + "c" * len(datasets) + "}",
        r"\hline",
        " & ".join(["Policy"] + datasets) + r" \\",
        r"\hline",
    ]
    for policy in policies:
        cells = [policy]
        for dataset in datasets:
            times = times_by_combo.get((dataset, policy))
            if times is None:
                cells.append("-")
            else:
                mean = statistics.mean(times)
                std = statistics.stdev(times) if len(times) > 1 else 0.0
                cells.append(_wrap(f"{mean:.2f} ({std:.2f})"))
        tex_lines.append(" & ".join(cells) + r" \\")
    tex_lines += [r"\hline", r"\end{tabular}"]

    tex_path = log_dir / f"{stem}.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"
    tag = "inst1_128k"

    log_dir = base_path / "evaluation" / tag / "logs"
    main(
        log_dir=log_dir,
        base_path=str(base_path),
    )
