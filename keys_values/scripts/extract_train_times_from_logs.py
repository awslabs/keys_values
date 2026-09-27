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

_TRAIN_RE = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*iter\s+(\d+)\s*\|.*\|\s*iter time:\s*([\d.]+)\s*(ms|s)"
)
_VALID_RE = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*iter\s+(\d+)\s*\|.*\|\s*val_time:\s*([\d.]+)\s*(ms|s)"
)


def _find_log_files(log_dir: Path) -> List[Path]:
    result = []
    if log_dir.exists():
        path = log_dir / "gpu0.log"
        if path.exists():
            result.append(path)
        for child in log_dir.iterdir():
            if child.name.startswith("resume"):
                path = child / "gpu0.log"
                if path.exists():
                    result.append(path)
    return result


def _parse_logs(
    log_files: List[Path], mode: str, filter_epochs: Callable[[int], bool]
) -> List[tuple]:
    pattern = _TRAIN_RE if mode == "train" else _VALID_RE
    records = []
    for log_file in log_files:
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


def _for_output(paths: List[Path], len_base: int) -> List[str]:
    off = len_base + 1
    return [str(path)[off:] for path in paths]


def _print_ratios(ratios: List[Optional[float]]) -> str:
    return ", ".join(f"{((r - 1) * 100):.1f}" for r in ratios if r is not None)


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
    len_base = len(str(base_path))

    for dataset, policy in product(datasets, policies):
        base_dir = base_path / dataset / policy
        if not base_dir.exists():
            continue
        log_dir = base_dir / "logs"
        log_files = _find_log_files(log_dir)
        if not log_files:
            continue
        records = _parse_logs(log_files, mode, filter_epochs)
        print(f"({dataset}, {policy}): {len(records)} records from {_for_output(log_files, len_base)}")
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

    # CSV file
    csv_path = base_path / f"times_{mode}_{dataset_size}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "policy", "epoch", "iter", "time_secs"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    # LaTeX file
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
    print(f"\nWriting timing table to {tex_path}")
    tex_path.write_text("\n".join(tex_lines) + "\n")

    # Ratios
    if mode == "train":
        for policy in policies:
            if "2048" in policy:
                pol_1024 = policy.replace("2048", "1024")
                pol_128 = policy.replace("2048", "128")
                do_128 = pol_128 in policies
                ratios_2048_1024 = []
                ratios_2048_128 = []
                for dataset in datasets:
                    times = times_by_combo.get((dataset, policy))
                    if times is None:
                        ratios_2048_1024.append(None)
                        if do_128:
                            ratios_2048_128.append(None)
                    else:
                        mean_2048 = statistics.mean(times)
                        times = times_by_combo.get((dataset, pol_1024))
                        if times is None:
                            ratios_2048_1024.append(None)
                        else:
                            mean_1024 = statistics.mean(times)
                            ratios_2048_1024.append(mean_1024 / mean_2048)
                        if do_128:
                            times = times_by_combo.get((dataset, pol_128))
                            if times is None:
                                ratios_2048_128.append(None)
                            else:
                                mean_128 = statistics.mean(times)
                                ratios_2048_128.append(mean_128 / mean_2048)
                num_1024 = sum(r is not None for r in ratios_2048_1024)
                if num_1024 > 0:
                    print(f"{policy}: 1k/2k  = [{_print_ratios(ratios_2048_1024)}]")
                if do_128:
                    num_128 = sum(r is not None for r in ratios_2048_128)
                    if num_128 > 0:
                        print(f"{policy}: 128/2k = [{_print_ratios(ratios_2048_128)}]")


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"

    dataset_size = "64k"
    # dataset_size = "128k"
    is_rerun = True
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
    if dataset_size == "64k":
        policies.extend(
            [
                "slr_4gpu_cs128_lr5",
                "h2o_4gpu_cs128_lr5",
                "h2onorm_4gpu_cs128_lr5",
                "h2oorig_4gpu_cs128_lr5",
            ]
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
