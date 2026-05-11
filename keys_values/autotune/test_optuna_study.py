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
"""Smoke test for optuna_study.py, runnable locally without GPU.

Run with:
    ~/virtenvs/keysvals_optuna/bin/python \
        keys_values/autotune/test_optuna_study.py
"""

import random
import sys
import tempfile
import os
import threading

# Make the package importable when run from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from keys_values.autotune.optuna_study import (
    OptunaStudyConfig,
    ask_optuna_trial,
    create_optuna_study,
    open_optuna_study,
    is_valid_combination,
    tell_optuna_trial,
    _constraints_func,
)

# Search space matching VARIABLE_CHOICES in autotune_full.py
VARIABLE_CHOICES = {
    "kv_cache:buffer_name": ("default", "torch-quantized8", "torch-quantized4"),
    "kv_cache:cache_length": (32768, 34816, 36864, 40960),
    "kv_cache:chunk_size": (1024, 2048, 4096),
    "kv_cache:cpu_offload": (True, False),
    "grad:layers_per_cell": (1, 2),
    "grad:chunks_per_cell_multiplier": (0.75, 1, 1.25, 1.5),
    "grad:layercp_qname": ("default", "torch-quantized8"),
    "grad:cachecp_qname": ("default", "torch-quantized8"),
}


def mock_eval_autotune_metrics(variables):
    """Mock that returns realistic-looking metrics based on configuration."""
    oom = variables.get("kv_cache:cpu_offload") and random.random() < 0.1
    rex = random.random() < 0.05
    if oom:
        return {"out_of_memory": True, "runtime_exception": False}
    if rex:
        return {"out_of_memory": False, "runtime_exception": True}
    base_train = 10.0 + variables["kv_cache:cache_length"] / 10000.0
    base_eval = 5.0 + variables["kv_cache:chunk_size"] / 2000.0
    return {
        "out_of_memory": False,
        "runtime_exception": False,
        "time_train": base_train + random.uniform(-0.5, 0.5),
        "time_eval": base_eval + random.uniform(-0.3, 0.3),
    }


def test_single_process(n_trials=12, sampler="nsgaii"):
    print(f"\n=== test_single_process (sampler={sampler}, n_trials={n_trials}) ===")
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        storage_path = f.name
    try:
        config = OptunaStudyConfig(
            storage_path=storage_path,
            study_name="smoke_test",
            sampler_name=sampler,
        )
        study = create_optuna_study(config)

        for i in range(n_trials):
            variables, trial = ask_optuna_trial(study, VARIABLE_CHOICES)
            assert is_valid_combination(variables), f"Invalid combo: {variables}"
            results = mock_eval_autotune_metrics(variables)
            tell_optuna_trial(study, trial, results)
            feasible = not results.get("out_of_memory") and not results.get(
                "runtime_exception"
            )
            t_train = f"{results['time_train']:.2f}" if feasible else "N/A"
            t_eval = f"{results['time_eval']:.2f}" if feasible else "N/A"
            print(
                f"  trial {i:2d}: feasible={feasible} time_train={t_train} time_eval={t_eval}"
            )

        from optuna.trial import TrialState

        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        feasible_trials = [
            t
            for t in completed
            if not t.user_attrs.get("out_of_memory")
            and not t.user_attrs.get("runtime_exception")
        ]
        pareto = study.best_trials
        print(
            f"  Completed {n_trials} evaluations: "
            f"{len(feasible_trials)} feasible, {len(pareto)} on Pareto front "
            f"({len(study.trials)} total trials including FAIL)."
        )
        assert len(completed) == n_trials
    finally:
        os.unlink(storage_path)
    print("  PASSED")


def test_multi_worker(n_workers=4, trials_per_worker=5):
    """Mirrors the actual usage: one create_optuna_study call before workers start,
    then each worker independently opens the same study via open_optuna_study."""
    print(
        f"\n=== test_multi_worker ({n_workers} workers x {trials_per_worker} trials) ==="
    )
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        storage_path = f.name
    try:
        config = OptunaStudyConfig(
            storage_path=storage_path,
            study_name="multi_worker",
            sampler_name="nsgaii",
        )
        # Simulate setup_internal: create the study once before spawning workers.
        create_optuna_study(config)

        trial_numbers_seen = []
        lock = threading.Lock()
        errors = []

        def worker(rank):
            try:
                # Simulate main: each rank opens (connects to) the existing study.
                study = open_optuna_study(config)
                for _ in range(trials_per_worker):
                    variables, trial = ask_optuna_trial(study, VARIABLE_CHOICES)
                    assert is_valid_combination(
                        variables
                    ), f"Rank {rank}: invalid combo"
                    results = mock_eval_autotune_metrics(variables)
                    tell_optuna_trial(study, trial, results)
                    with lock:
                        trial_numbers_seen.append(trial.number)
            except Exception as e:
                with lock:
                    errors.append(f"rank {rank}: {e}")

        threads = [threading.Thread(target=worker, args=(r,)) for r in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            raise AssertionError("Worker errors:\n" + "\n".join(errors))

        total_expected = n_workers * trials_per_worker
        assert (
            len(trial_numbers_seen) == total_expected
        ), f"Expected {total_expected} reported trials, got {len(trial_numbers_seen)}"
        # Every trial number must be unique: no two workers evaluated the same trial.
        assert len(set(trial_numbers_seen)) == len(
            trial_numbers_seen
        ), f"Duplicate trial numbers found: {sorted(trial_numbers_seen)}"
        print(
            f"  All {total_expected} trials have unique numbers: {sorted(trial_numbers_seen)}"
        )
    finally:
        os.unlink(storage_path)
    print("  PASSED")


def test_constraint_encoding():
    print("\n=== test_constraint_encoding ===")
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        storage_path = f.name
    try:
        config = OptunaStudyConfig(
            storage_path=storage_path, study_name="constraint_test"
        )
        study = create_optuna_study(config)

        variables, trial = ask_optuna_trial(study, VARIABLE_CHOICES)
        tell_optuna_trial(
            study, trial, {"out_of_memory": True, "runtime_exception": False}
        )
        frozen = study.trials[-1]
        constraint_val = _constraints_func(frozen)
        assert constraint_val[0] > 0, "OOM trial should be infeasible (constraint > 0)"

        variables, trial = ask_optuna_trial(study, VARIABLE_CHOICES)
        tell_optuna_trial(
            study,
            trial,
            {
                "out_of_memory": False,
                "runtime_exception": False,
                "time_train": 8.0,
                "time_eval": 4.0,
            },
        )
        frozen = study.trials[-1]
        constraint_val = _constraints_func(frozen)
        assert constraint_val[0] < 0, "Feasible trial should have constraint <= 0"
        print("  constraint encoding OK")
    finally:
        os.unlink(storage_path)
    print("  PASSED")


def test_resume():
    print("\n=== test_resume (workers see trials from previous run) ===")
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        storage_path = f.name
    try:
        config = OptunaStudyConfig(storage_path=storage_path, study_name="resume_test")

        # First run: create study and evaluate one trial.
        study1 = create_optuna_study(config)
        variables, trial = ask_optuna_trial(study1, VARIABLE_CHOICES)
        tell_optuna_trial(
            study1,
            trial,
            {
                "out_of_memory": False,
                "runtime_exception": False,
                "time_train": 9.0,
                "time_eval": 3.5,
            },
        )
        assert len(study1.trials) == 1

        # Second run: open the same study — workers see the prior trial.
        study2 = open_optuna_study(config)
        assert (
            len(study2.trials) == 1
        ), f"Expected 1 trial after resume, got {len(study2.trials)}"
        print("  resumed study sees prior trials")
    finally:
        os.unlink(storage_path)
    print("  PASSED")


def test_custom_variable_choices():
    print("\n=== test_custom_variable_choices ===")
    custom_choices = {
        "kv_cache:buffer_name": ("torch-quantized8",),
        "kv_cache:cache_length": (32768, 40960),
        "kv_cache:chunk_size": (1024, 2048),
        "kv_cache:cpu_offload": (False,),
        "grad:layers_per_cell": (1, 2),
        "grad:chunks_per_cell_multiplier": (1.0, 1.5),
        "grad:layercp_qname": ("default",),
        "grad:cachecp_qname": ("default",),
    }
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        storage_path = f.name
    try:
        config = OptunaStudyConfig(
            storage_path=storage_path,
            study_name="custom_choices",
            variable_choices=custom_choices,
        )
        study = create_optuna_study(config)
        effective_choices = config.variable_choices
        for _ in range(5):
            variables, trial = ask_optuna_trial(study, effective_choices)
            assert variables["kv_cache:buffer_name"] == "torch-quantized8"
            assert variables["kv_cache:cpu_offload"] is False
            tell_optuna_trial(
                study,
                trial,
                {
                    "out_of_memory": False,
                    "runtime_exception": False,
                    "time_train": 7.0,
                    "time_eval": 3.0,
                },
            )
        print("  all custom choices respected")
    finally:
        os.unlink(storage_path)
    print("  PASSED")


if __name__ == "__main__":
    random.seed(42)
    test_constraint_encoding()
    test_single_process(n_trials=12, sampler="nsgaii")
    test_single_process(n_trials=8, sampler="tpe")
    test_single_process(n_trials=6, sampler="random")
    test_resume()
    test_custom_variable_choices()
    test_multi_worker(n_workers=4, trials_per_worker=5)
    print("\nAll tests passed.")
