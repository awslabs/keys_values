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

from keys_values.evaluation.tasks import EvaluationTasks


def main(out_dir: Path, model_type: str):
    total_removed = 0
    eval_tasks = EvaluationTasks(out_dir, model_type)
    for task_name, incomplete_file_paths in eval_tasks.eval_result_files(
        return_incompletes=True,
    ):
        print(f"{task_name}: Removing {len(incomplete_file_paths)} lock files")
        for path in incomplete_file_paths:
            path.unlink()
        total_removed += len(incomplete_file_paths)
    print(f"\nRemoved {total_removed} lock files in total")


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"
    out_dirs = [
        "helmet_nq_64k/slr_4gpu_cs2048_lr5",
        "helmet_trivia_qa_64k/lr_4gpu_cs2048_lr5",
        "helmet_trivia_qa_64k/h2o_4gpu_cs2048_lr5",
    ]
    model_type = "lora"
    for out_dir in out_dirs:
        main(base_path / out_dir, model_type)
