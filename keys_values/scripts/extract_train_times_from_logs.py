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
from typing import Callable, List, Optional


_TRAIN_RE = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*iter\s+(\d+)\s*\|.*\|\s*iter time:\s*([\d.]+)\s*(ms|s)\b"
)
_VALID_RE = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*iter\s+(\d+)\s*\|.*\|\s*val_time:\s*([\d.]+)\s*(ms|s)\b"
)


def _find_log_file(log_dir: Path) -> Optional[Path]:
    resume_candidates = []
    if log_dir.exists():
        for child in log_dir.iterdir():
            m = re.fullmatch(r"resume(\d+)", child.name)
            if m and (child / "gpu0.log").exists():
                resume_candidates.append((int(m.group(1)), child / "gpu0.log"))
    if resume_candidates:
        return max(resume_candidates, key=lambda x: x[0])[1]
    base_log = log_dir / "gpu0.log"
    return base_log if base_log.exists() else None


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
        log_file = _find_log_file(log_dir)
        if log_file is None:
            continue
        print(f"({dataset}, {policy}): {log_file}")
        records = _parse_log(log_file, mode, filter_epochs)
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
    tex_path.write_text("\n".join(tex_lines) + "\n")


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"

    dataset_size = "64k"
    # dataset_size = "128k"
    is_rerun = False
    if is_rerun:
        base_path = base_path / "rerun"
    datasets = [
        f"helmet_nq_{dataset_size}",
        f"helmet_trivia_qa_{dataset_size}",
        f"helmet_hotpot_qa_{dataset_size}",
        f"helmet_pop_qa_{dataset_size}",
    ]
    policies = [
        "lr_4gpu_cs2048_lr5",
        "slr_4gpu_cs2048_lr5",
        "h2o_4gpu_cs2048_lr5",
        "h2onorm_4gpu_cs2048_lr5",
        "h2oorig_4gpu_cs2048_lr5",
        "lr_4gpu_cs1024_lr5",
        "slr_4gpu_cs1024_lr5",
        "h2o_4gpu_cs1024_lr5",
        "h2onorm_4gpu_cs1024_lr5",
        "h2oorig_4gpu_cs1024_lr5",
    ]
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
