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
"""Optuna study configuration and helpers for autotune_full / autotune_lora.

``optuna`` is an optional dependency.  Importing this module succeeds even
when ``optuna`` is not installed; a descriptive ``ImportError`` is raised only
when a function that actually uses Optuna is called.

Multi-process setup
-------------------
Every rank launched by ``fabric.launch`` runs its own independent trial loop.
All ranks open the **same** ``JournalFileBackend`` log file; the storage uses
local filesystem locking to serialise concurrent ``ask`` / ``tell`` calls, so
each rank receives a distinct configuration from the sampler without any
network communication.

Constrained multi-objective optimisation
-----------------------------------------
The two objectives are ``time_train`` and ``time_eval`` (both minimised).
The feasibility constraints ``out_of_memory=False`` and
``runtime_exception=False`` are encoded via the sampler's ``constraints_func``.
Infeasible trials are still reported with ``study.tell`` (using large fallback
values) so the sampler can learn which configurations tend to violate
constraints.
"""

# All annotations are treated as strings at runtime, so references to optuna
# types in signatures do not trigger an ImportError when optuna is absent.
from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Dict, Optional, Tuple, Callable, Sequence, Literal

try:
    import optuna
    from optuna.samplers import NSGAIISampler, RandomSampler, TPESampler
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    from optuna.trial import TrialState

    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


# Names of the user attributes written by tell_optuna_trial and read back by
# _constraints_func.  These must match across ask/tell calls.
_ATTR_OOM = "out_of_memory"
_ATTR_RUNTIME_EX = "runtime_exception"

# Fallback objective values used when a trial fails before producing metrics
# (e.g. OOM before timing starts).  Using a large finite value rather than
# infinity keeps the sampler numerics well-behaved.
_FALLBACK_TIME = 1e9

# Maximum number of consecutive invalid combinations before giving up.
MAX_SUGGESTION_TRIES = 50


def kwargs_not_none(obj: Any, prefix: str) -> Dict[str, Any]:
    p_len = len(prefix)
    return {
        k[p_len:]: v
        for k, v in dataclasses.asdict(obj).items()
        if v is not None and k.startswith(prefix)
    }


def require_optuna() -> None:
    if not _OPTUNA_AVAILABLE:
        raise ImportError(
            "The 'optuna' package is required for Optuna-guided search but is "
            "not installed in the current environment.  Install it with:\n\n"
            "    pip install optuna\n"
        )


@dataclasses.dataclass
class OptunaArgs:
    name: Literal["nsgaii", "tpe", "random"] = "nsgaii"
    # Parameters for nsgaii
    nsg_population_size: Optional[int] = None
    nsg_pmutation_prob: Optional[float] = None
    nsg_pcrossover: Optional[Any] = (None,)
    nsg_pcrossover_prob: Optional[float] = None
    nsg_pswapping_prob: Optional[float] = None
    # Parameters for tpe
    tpe_consider_prior: Optional[bool] = None
    tpe_prior_weight: Optional[float] = None
    tpe_consider_magic_clip: Optional[bool] = None
    tpe_consider_endpoints: Optional[bool] = None
    tpe_n_startup_trials: Optional[int] = None
    tpe_n_ei_candidates: Optional[int] = None
    tpe_gamma: Optional[Callable[[int], int]] = None
    tpe_weights: Optional[bool] = None
    tpe_multivariate: Optional[bool] = None
    tpe_group: Optional[bool] = None
    tpe_warn_independent_sampling: Optional[bool] = None
    tpe_constant_liar: Optional[bool] = True

    def __post_init__(self):
        if self.name not in ("nsgaii", "tpe", "random"):
            raise ValueError(
                f"name = {self.name}, must be in ('nsgaii', 'tpe', 'random')"
            )

    def get_sampler(
        self,
        constraints_func: Optional[
            Callable[[optuna.trial.FrozenTrial], Sequence[float]]
        ] = None,
        seed: Optional[int] = None,
    ) -> optuna.samplers.BaseSampler:
        require_optuna()
        extra_kwargs = {"constraints_func": constraints_func}
        if self.name == "nsgaii":
            args_class = NSGAIISampler
        elif self.name == "tpe":
            args_class = TPESampler
        else:
            args_class = RandomSampler
            extra_kwargs = dict()
        return args_class(
            seed=seed,
            **self.kwargs_not_none(),
            **extra_kwargs,
        )

    def kwargs_not_none(self) -> Dict[str, Any]:
        prefix = self.name[:3] + "_"
        return kwargs_not_none(self, prefix)


# ---------------------------------------------------------------------------
# Sampler / constraint helpers
# ---------------------------------------------------------------------------


def _constraints_func(trial: optuna.trial.FrozenTrial):
    """Return positive value when feasibility constraints are violated.

    The convention used by NSGAIISampler / TPESampler is: a return value <= 0
    means feasible; > 0 means infeasible.
    """
    oom = trial.user_attrs.get(_ATTR_OOM, False)
    rex = trial.user_attrs.get(_ATTR_RUNTIME_EX, False)
    return [1.0 if (oom or rex) else -1.0]


def _make_sampler(
    sampler_args: OptunaArgs,
    seed: Optional[int] = None,
) -> optuna.samplers.BaseSampler:
    """Instantiate the requested sampler with constraint support where available."""
    with warnings.catch_warnings():
        # constraints_func is marked experimental in the installed Optuna
        # version; suppress the warning since we intentionally use it.
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        return sampler_args.get_sampler(
            constraints_func=_constraints_func,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class OptunaStudyConfig:
    """Configuration that describes how to create / resume an Optuna study.

    Pass an instance to ``setup`` / ``setup_internal`` to replace the default
    random search with Optuna-guided optimisation.

    Arguments:
        storage_path: Path to the JournalFileBackend log file.  All worker
            processes on the same host must use the **same** path so that they
            share the study state.  The file is created if it does not exist.
        study_name: Name of the Optuna study.  Re-using the same name and
            ``storage_path`` continues an existing study rather than starting
            fresh.
        sampler_args: Which sampler to use, and its arguments.
        variable_choices: Search-space definition as a mapping from parameter
            name to a tuple of candidate values.  When ``None`` (the default),
            ``VARIABLE_CHOICES`` defined in ``autotune_full`` is used.  You
            can pass a subset of keys or different tuples to restrict or widen
            individual search ranges.
    """

    storage_path: str
    study_name: str
    sampler_args: OptunaArgs
    variable_choices: Optional[Dict[str, tuple]] = None


# ---------------------------------------------------------------------------
# Study creation
# ---------------------------------------------------------------------------


def create_optuna_study(
    config: OptunaStudyConfig,
    seed: Optional[int] = None,
) -> optuna.Study:
    """Create an Optuna study and record it in the journal storage.

    This is called **once** before workers are launched (i.e. in
    ``setup_internal``, before ``fabric.launch``).  It writes the study record
    to the ``JournalFileBackend`` log file so that all workers find a
    pre-existing study when they later call :func:`open_optuna_study`.

    The study optimises two objectives (``time_train``, ``time_eval``) jointly,
    both to be minimised.  Feasibility constraints (OOM, runtime exception) are
    handled by ``_constraints_func`` inside the sampler.

    Raises:
        ImportError: if ``optuna`` is not installed.
        RuntimeError: if a study with ``config.study_name`` already exists in
            the storage (use a different name or delete the log file).
    """
    require_optuna()
    storage = JournalStorage(JournalFileBackend(file_path=config.storage_path))
    sampler = _make_sampler(config.sampler_args, seed=seed)

    study = optuna.create_study(
        storage=storage,
        study_name=config.study_name,
        sampler=sampler,
        directions=["minimize", "minimize"],
        load_if_exists=False,
    )
    study.set_metric_names(["time_train", "time_eval"])
    return study


def open_optuna_study(
    config: OptunaStudyConfig,
    seed: Optional[int] = None,
) -> optuna.Study:
    """Open an existing study from storage for use inside a worker (``main``).

    This is called by each worker **after** the study has been created by
    :func:`create_optuna_study` in the parent process.  A fresh sampler
    instance is built so that each worker has its own in-process sampler state
    (Optuna samplers are not designed to be shared across processes).

    Raises:
        ImportError: if ``optuna`` is not installed.
    """
    require_optuna()
    storage = JournalStorage(JournalFileBackend(file_path=config.storage_path))
    sampler = _make_sampler(config.sampler_args, seed=seed)
    return optuna.load_study(
        storage=storage,
        study_name=config.study_name,
        sampler=sampler,
    )


# ---------------------------------------------------------------------------
# Per-trial ask / tell helpers
# ---------------------------------------------------------------------------


def suggest_variables(
    trial: optuna.Trial,
    variable_choices: Dict[str, tuple],
) -> Dict[str, Any]:
    """Suggest one value per variable using ``suggest_categorical``."""
    return {
        name: trial.suggest_categorical(name, list(choices))
        for name, choices in variable_choices.items()
    }


def is_valid_combination(variables: Dict[str, Any]) -> bool:
    """Return ``False`` for known-invalid parameter combinations.

    cpu_offload requires a non-default buffer (quantized or similar), because
    the default buffer does not support offloading.
    """
    if (
        variables.get("kv_cache:cpu_offload")
        and variables.get("kv_cache:buffer_name") == "default"
    ):
        return False
    return True


def ask_optuna_trial(
    study: optuna.Study,
    variable_choices: Dict[str, tuple],
) -> Tuple[Dict[str, Any], optuna.Trial]:
    """Ask Optuna for the next trial for this rank.

    Each rank calls this independently.  The shared ``JournalFileBackend``
    storage serialises concurrent ``ask`` calls via file locking, so every rank
    receives a distinct trial suggested by the sampler.

    Invalid combinations (see ``is_valid_combination``) are rejected with
    ``TrialState.FAIL`` and a new trial is requested immediately, up to
    ``_MAX_SUGGESTION_TRIES`` times.

    Returns:
        ``(variables, trial)`` — the suggested parameter dict and the live
        :class:`~optuna.Trial` to be passed to :func:`tell_optuna_trial` after
        evaluation.

    Raises:
        ImportError: if ``optuna`` is not installed.
    """
    require_optuna()
    for _ in range(MAX_SUGGESTION_TRIES):
        trial = study.ask()
        variables = suggest_variables(trial, variable_choices)
        if is_valid_combination(variables):
            return variables, trial
        study.tell(trial, state=TrialState.FAIL)
    raise RuntimeError(
        f"Could not sample a valid configuration after {MAX_SUGGESTION_TRIES} tries."
    )


def tell_optuna_trial(
    study: optuna.Study,
    trial: optuna.Trial,
    results: Dict[str, Any],
) -> None:
    """Report evaluation results back to Optuna.

    Should be called only on ``fabric.global_rank == 0`` (i.e. where ``trial``
    is not ``None``).

    The feasibility-constraint attributes are written to the trial before
    calling ``study.tell`` so that ``_constraints_func`` can read them when the
    sampler queries this trial for future suggestions.

    If the trial ended with an OOM error or a runtime exception before timing
    data was collected, ``_FALLBACK_TIME`` is used as the objective value so
    that ``study.tell`` can still record the trial.

    Raises:
        ImportError: if ``optuna`` is not installed.
    """
    require_optuna()
    oom = bool(results.get("out_of_memory", False))
    rex = bool(results.get("runtime_exception", False))
    trial.set_user_attr(_ATTR_OOM, oom)
    trial.set_user_attr(_ATTR_RUNTIME_EX, rex)

    time_train = results.get("time_train", _FALLBACK_TIME)
    time_eval = results.get("time_eval", _FALLBACK_TIME)
    study.tell(trial, values=[time_train, time_eval])
