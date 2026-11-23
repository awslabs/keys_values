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
from itertools import product
from pathlib import Path
from typing import Callable, List

from keys_values.scripts.cleanup_evaluation import datasets_and_cases

LOG_FILENAME = "gpu0.log"

_TRAIN_RE = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*iter\s+(\d+)\s*\|.*\|\s*iter time:\s*([\d.]+)\s*(ms|s)\b"
)
_VALID_RE = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*iter\s+(\d+)\s*\|.*\|\s*val_time:\s*([\d.]+)\s*(ms|s)\b"
)


def _find_log_files(log_dir: Path) -> List[Path]:
    result = []
    if log_dir.exists():
        for child in log_dir.iterdir():
            m = re.fullmatch(r"resume(\d+)", child.name)
            base_log = child / LOG_FILENAME
            if m and base_log.exists():
                result.append(base_log)
        base_log = log_dir / LOG_FILENAME
        if base_log.exists():
            result.append(base_log)
    return result


def _parse_log(
    log_file: Path, mode: str, filter_epochs: Callable[[int], bool]
) -> List[tuple]:
    pattern = _TRAIN_RE if mode == "train" else _VALID_RE
    records = []
    with log_file.open() as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            if not filter_epochs(epoch):
                continue
            iter_val = int(m.group(2))
            time_val = float(m.group(3))
            if m.group(4) == "ms":
                time_val /= 1000.0
            records.append((epoch, iter_val, time_val))
    return records


def _wrap(s: str) -> str:
    return "{\\small\\!" + s + "}"


def _last_two(p: Path) -> str:
    p_name = p.parent.name
    return (p_name + "/" if p_name != "logs" else "") + p.name


def main(
    mode: str,
    dataset_size: str,
    datasets: List[str],
    policies: List[str],
    base_path: Path,
    filter_epochs: Callable[[int], bool],
):
    all_rows = []
    times_by_combo = {}  # (dataset, policy) -> [time_secs, ...]

    for dataset, policy in product(datasets, policies):
        base_dir = base_path / dataset / policy
        log_dir = base_dir / "logs"
        if not base_dir.exists():
            continue
        log_files = _find_log_files(log_dir)
        if not log_files:
            continue
        print(f"({dataset}, {policy}): {[_last_two(p) for p in log_files]}")
        records = [
            entry
            for log_file in log_files
            for entry in _parse_log(log_file, mode, filter_epochs)
        ]
        times = []
        for epoch, iter_val, time_secs in records:
            all_rows.append(
                {
                    "dataset": dataset,
                    "policy": policy,
                    "epoch": epoch,
                    "iter": iter_val,
                    "time_secs": time_secs,
                }
            )
            times.append(time_secs)
        if times:
            times_by_combo[(dataset, policy)] = times

    csv_path = base_path / f"times_{mode}_{dataset_size}.csv"
    print(f"\nWriting {csv_path}")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "policy", "epoch", "iter", "time_secs"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

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

    tex_path = base_path / f"times_{mode}_{dataset_size}.tex"
    print(f"\nWriting {tex_path}")
    tex_path.write_text("\n".join(tex_lines) + "\n")


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"

    dataset_size = "128k"
    is_rerun = False
    is_baseline = False
    is_base_model = False
    extra_data = True
    is_ruler = False
    filter_dataset = None
    filter_case = None
    chunk_sizes = None

    if is_rerun:
        base_path = base_path / "rerun"
    elif is_baseline:
        base_path = base_path / "baseline"
    elif is_base_model:
        base_path = base_path / "basemod"
    datasets, policies = datasets_and_cases(
        dataset_size,
        extra_data,
        is_baseline,
        is_base_model,
        is_ruler,
        filter_dataset=filter_dataset,
        filter_case=filter_case,
        chunk_sizes=chunk_sizes,
    )

    # Skip epoch 0 (warm-up)
    filter_epochs = lambda epoch: epoch > 0

    for mode in ("train", "valid"):
        main(
            mode,
            dataset_size,
            datasets,
            policies,
            base_path,
            filter_epochs,
        )
