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
"""Visualize an autotune Optuna study.

Usage
-----
Activate the keysvals_optuna virtual environment, then run from the repo root::

    python keys_values/autotune/visualize_study.py \\
        --storage path/to/journal.log \\
        --study   autotune \\
        [--out    results/]      # directory for HTML output; default: current dir
        [--no-browser]           # write files but do not open browser

Plots produced
--------------
pareto_front.html
    Scatter plot of (time_train, time_eval) for every completed trial.
    Three series are distinguished:

    * **Pareto-front / Best** (red scale) — feasible, non-dominated trials.
    * **Feasible** (blue scale) — feasible but dominated by at least one
      Pareto-front trial.
    * **Infeasible** (grey) — trials that ended with out-of-memory or a
      runtime exception.

    Feasibility is read directly from the ``constraints`` system attribute
    written by NSGAIISampler / TPESampler (via ``constraints_func``), so no
    extra arguments are needed.

hypervolume_history.html
    Hypervolume indicator over trial index, a scalar summary of how the
    Pareto front improves over time.  Only feasible trials contribute.

param_importances.html (x2)
    Fanova-based hyperparameter importance for each objective separately.
    Shows which parameters most affect ``time_train`` and ``time_eval``.
"""

from __future__ import annotations

import argparse
import os
import webbrowser
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def _load_study(storage_path: str, study_name: str) -> optuna.Study:
    storage = JournalStorage(JournalFileBackend(file_path=storage_path))
    return optuna.load_study(study_name=study_name, storage=storage)


def _save_and_maybe_open(fig, path: Path, open_browser: bool) -> None:
    fig.write_html(str(path))
    print(f"  Written: {path}")
    if open_browser:
        webbrowser.open(path.as_uri())


def visualize(
    storage_path: str, study_name: str, out_dir: Path, open_browser: bool
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    study = _load_study(storage_path, study_name)

    from optuna.trial import TrialState

    completed = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    print(
        f"Study '{study_name}': {len(study.trials)} total trials, "
        f"{len(completed)} completed."
    )

    # ------------------------------------------------------------------
    # Pareto front
    # Constraints are stored in trial.system_attrs['constraints'] by the
    # sampler's after_trial hook, so plot_pareto_front picks them up and
    # draws three series (infeasible / feasible / Pareto-front) automatically.
    # ------------------------------------------------------------------
    print("\nGenerating pareto_front.html ...")
    fig = optuna.visualization.plot_pareto_front(
        study,
        include_dominated_trials=True,
    )
    _save_and_maybe_open(fig, out_dir / "pareto_front.html", open_browser)

    # ------------------------------------------------------------------
    # Hypervolume history — requires a reference point slightly worse than
    # the worst observed values on each objective.
    # ------------------------------------------------------------------
    feasible = [
        t
        for t in completed
        if all(x <= 0.0 for x in (t.system_attrs.get("constraints") or []))
    ]
    if len(feasible) >= 2:
        print("Generating hypervolume_history.html ...")
        ref_point = [
            max(t.values[i] for t in feasible) * 1.1
            for i in range(len(study.directions))
        ]
        fig = optuna.visualization.plot_hypervolume_history(
            study, reference_point=ref_point
        )
        _save_and_maybe_open(fig, out_dir / "hypervolume_history.html", open_browser)
    else:
        print("Skipping hypervolume_history.html (fewer than 2 feasible trials).")

    # ------------------------------------------------------------------
    # Parameter importances — one plot per objective.
    # ------------------------------------------------------------------
    if len(feasible) >= 4:
        for obj_idx, obj_name in enumerate(
            study.metric_names or ["time_train", "time_eval"]
        ):
            print(f"Generating param_importances_{obj_name}.html ...")
            fig = optuna.visualization.plot_param_importances(
                study,
                target=lambda t, i=obj_idx: t.values[i],
                target_name=obj_name,
            )
            _save_and_maybe_open(
                fig, out_dir / f"param_importances_{obj_name}.html", open_browser
            )
    else:
        print("Skipping param_importances (fewer than 4 feasible trials).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize an autotune Optuna study.")
    parser.add_argument(
        "--storage", required=True, help="Path to the JournalFileBackend log file."
    )
    parser.add_argument("--study", required=True, help="Name of the Optuna study.")
    parser.add_argument(
        "--out",
        default=".",
        help="Output directory for HTML files (default: current dir).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Write HTML files but do not open a browser.",
    )
    args = parser.parse_args()
    visualize(
        storage_path=args.storage,
        study_name=args.study,
        out_dir=Path(args.out),
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
