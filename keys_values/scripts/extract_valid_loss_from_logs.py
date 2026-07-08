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
import re
from itertools import product
from pathlib import Path


def main(
    base_path: Path,
    dataset: str,
    policy: str,
    patience: int,
):
    assert patience > 0
    log_path = base_path / dataset / policy / "logs"

    # Collect log files: gpu0.log and resume{N}/gpu0.log for positive integer N
    log_files = []
    direct = log_path / "gpu0.log"
    if direct.exists():
        log_files.append(direct)
    for resume_dir in sorted(log_path.glob("resume*")):
        if resume_dir.is_dir():
            suffix = resume_dir.name[len("resume"):]
            if suffix.isdigit() and int(suffix) > 0:
                f = resume_dir / "gpu0.log"
                if f.exists():
                    log_files.append(f)

    if not log_files:
        print(f"{dataset}, {policy}: No logs")
        return

    # Parse iter -> val_loss from all log files (last occurrence wins on collision)
    iter_pattern = re.compile(r'iter\s+(\d+)\s+\|.*val_loss:\s*([\d.]+)')
    init_pattern = re.compile(r'Initial evaluation\s+\|.*val_loss:\s*([\d.]+)')
    records: dict[int, float] = {}
    for log_file in log_files:
        with open(log_file) as fh:
            for line in fh:
                m = iter_pattern.search(line)
                if m:
                    records[int(m.group(1))] = float(m.group(2))
                    continue
                m = init_pattern.search(line)
                if m:
                    records[0] = float(m.group(1))

    if not records:
        print(f"{dataset}, {policy}: No logs")
        return

    pairs = sorted(records.items())
    iters = [p[0] for p in pairs]
    val_losses = [p[1] for p in pairs]
    n = len(iters)

    # (iter_0, val_loss_0): global minimum
    idx_0 = min(range(n), key=lambda i: val_losses[i])
    iter_0, val_loss_0 = iters[idx_0], val_losses[idx_0]

    # patience-based i_star, linear via running minimum
    cnt = [0] * n
    running_min = val_losses[0]
    for i in range(1, n):
        if val_losses[i] < running_min:
            running_min = val_losses[i]
            cnt[i] = 0
        else:
            cnt[i] = cnt[i - 1] + 1

    i_star = n
    for i, v in enumerate(cnt):
        if v >= patience:
            i_star = i
            break

    # (iter_1, val_loss_1): minimum of val_loss[:i_star] (exclusive)
    prefix = val_losses[:i_star]
    idx_1 = min(range(len(prefix)), key=lambda i: prefix[i])
    iter_1, val_loss_1 = iters[idx_1], prefix[idx_1]

    print(
        f"{dataset}, {policy}: "
        f"0: step-{iter_0:06d} ({val_loss_0:.2f}) | "
        f"1: step-{iter_1:06d} ({val_loss_1:.2f})"
    )


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"

    patience = 5
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

    for dataset, policy in product(datasets, policies):
        main(base_path, dataset, policy, patience)
