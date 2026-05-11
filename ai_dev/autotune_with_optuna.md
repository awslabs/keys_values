# Write auto-tuning script `keys_values/autotune_full` with Claude Code

## Initial prompt

I'd like you to create a script for auto-tuning of certain parameters:

* Should run on a single GPU instance with a number of devices
* Each device runs evaluation jobs for configurations suggested by the tuner.
  An evaluation job returns metrics, which are to be optimized

I created a basic version in `~/git/keys_values/autotune/autotune_full.py`:

* Each evaluation job runs `eval_autotune_metrics`.
* `eval_autotune_metrics` returns a dictionary with keys
  "out_of_memory" (bool), "runtime_exception" (bool), "time_eval" (float),
  "time_train" (float). The goal of tuning is to find configurations for which
  `out_of_memory=False, runtime_exception=False` (hard constraint), but
  "time_train", "time_eval" values are minimized.
* `main` implements a loop over jobs run on a single device. The configurations
  for jobs are drawn at random in `sample_configuration`, so this script
  implements random search.
* `VARIABLE_CHOICES` defines the parameters to optimize over and their
  search ranges.

I would like the tuning script to use `Optuna` instead of random search.
Find the optuna repository in `~/git/optuna`.

* The setup is "multi-process optimization". Have a look at the tutorial
  page: `~/git/optuna/tutorial/10_key_features/004_distributed.py`. Decide
  between `JournalStorage` or `RDBStorage`.
* `main` runs independently on each device. Find a way for Optuna to
  orchestrate the search, and to communicate with these runners.
* The solution must not use communication over the network.
* The user should be able to specify the Optuna study, including which
  algorithms to use, and how the search space is defined. The variables
  are in `VARIABLE_CHOICES`, but ranges may be different.
* The goal is constrained optimization of "time_train", "time_eval", given
  that `out_of_memory=False, runtime_exception=False`.

Constraints:
* Your solution must not change `eval_autotune_metrics`. Please edit
  `main` to take suggestions from Optuna instead of sampling them at random,
  and add additional functions as needed.
* Put new code into `~/git/keys_values/autotune`.
* Use the virtual environment `~/virtenvs/keysvals_optuna` for running Python
  code. This runs locally, not on a GPU instance. If this is a problem, you
  may want to write a mock for `eval_autotune_metrics`.

### Answer from Claude Code:

All tests pass and the implementation is complete. Here's a summary of what was built:

New file: `keys_values/autotune/optuna_study.py`

Contains all Optuna-specific logic:

* `OptunaStudyConfig` — user-facing dataclass with storage_path, study_name, sampler_name ("nsgaii" / "tpe" / "random"), sampler_kwargs, and variable_choices (overrides VARIABLE_CHOICES when set).
* `create_optuna_study(config)` — creates/resumes a multi-objective study (directions=["minimize", "minimize"]) backed by JournalFileBackend. Chosen over RDBStorage because it requires no server setup, uses local file locking, and is safe for multiple processes on a single host.
* `ask_optuna_trial(fabric, study, variable_choices)` — only fabric.global_rank == 0 calls study.ask(). Invalid combinations (e.g. cpu_offload=True + buffer_name="default") are rejected with TrialState.FAIL and a new trial is immediately requested. The chosen variables dict is broadcast to all DDP ranks via dist.broadcast_object_list.
* `tell_optuna_trial(study, trial, results)` — writes out_of_memory / runtime_exception as trial user attributes (read back by _constraints_func), then calls study.tell(trial, values=[time_train, time_eval]). OOM/exception trials are still reported with a large fallback time so the sampler learns from them.
* `_constraints_func` — returns [1.0] (infeasible) when either OOM or runtime exception occurred, [-1.0] otherwise. Used by NSGAIISampler and TPESampler to steer away from infeasible regions.

Edited: `keys_values/autotune/autotune_full.py`

* Added `optuna_config: Optional[OptunaStudyConfig] = None` to setup, setup_internal, main.
* `main` creates the study if optuna_config is given, and in the eval loop calls ask_optuna_trial / tell_optuna_trial instead of (or falling back to) sample_configuration.
* `sample_configuration` now accepts `variable_choices` so it respects the same override as the Optuna path.
* Fixed `1,000,000 (tuple) → 1_000_000 (int)`.

Test file: `keys_values/autotune/test_optuna_study.py` — runnable locally without GPU using a mock of eval_autotune_metrics.


## Comment:

This looks great. Could you rewrite `optuna_study.py` in a way that would guard against
`optuna` not being installed in the Python environment? I'd like to make the dependence
on `optuna` optional.

If `keys_valued.autotune.autotune_full` or `keys_valued.autotune.autotune_lora` are
called with `optuna_config not None` and `optuna` is not installed, raise an exception,
asking the user to install `optuna`.

### Answer from Claude Code:

All checks pass. Here's what changed and why each piece is needed:

* `from __future__ import annotations` — makes every annotation a lazy string that Python never evaluates at import time. Without this, the optuna.Study, optuna.Trial, etc. in function signatures would raise a NameError at module import time when optuna is absent, even if the function is never called.

* `try/except ImportError` block — attempts to import optuna and its submodules, sets _OPTUNA_AVAILABLE = True/False. The names optuna, NSGAIISampler, JournalStorage, etc. are simply undefined in the module namespace when the import fails, which is fine as long as no code path reaches them without first passing through the guard.

* `_require_optuna()` guard — called at the top of create_optuna_study, ask_optuna_trial, and tell_optuna_trial (the three functions that actually touch Optuna objects). is_valid_combination and suggest_variables are pure Python and don't need the guard. _constraints_func and _make_sampler are only ever called from inside functions that already checked, so they don't need it either.

The error message includes `pip install optuna` so the user gets actionable guidance immediately.


## Comment:

I noted a potential issue. In `keys_values.autotune.autotune_full.main`, you call
`tell_optuna_trial(study, trial, results)` only if `fabric.global_rank == 0`. Why?
Optuna should receive results from all trials, no matter which rank runs them. Note
there is no accumulation of trial results over ranks. Each rank should run its own
trials, but Optuna should coordinate the search.

If I am wrong, please do explain.

### Answer from Claude Code:

You are wrong, and the current code is correct. Let me explain the structure.

fabric.launch(main, ...) with DDPStrategy spawns devices cooperative processes — they run the same configuration together. DDP splits the batch across ranks and synchronises gradients; all ranks must process an identical model configuration at each step. It is not possible for rank 0 to evaluate config A while rank 1 evaluates config B within one fabric.launch call.

So the execution unit for one Optuna trial is the entire set of devices ranks acting in concert:

fabric.launch(main, ...)   ← one trial at a time
├── rank 0: ask Optuna → broadcast config → eval → tell Optuna
├── rank 1: .............. receive config → eval (same config, DDP peer)
└── rank N: .............. receive config → eval (same config, DDP peer)
All ranks compute the same result; only rank 0 needs to report it once. Reporting from every rank would register devices duplicate trials for what was a single evaluation.

Multi-device parallelism in Optuna's sense — where multiple independent trials run simultaneously — would come from launching multiple independent setup calls (separate processes, each with its own fabric.launch), all pointing to the same storage_path. The JournalFileBackend then coordinates between those separate groups, each of which is internally DDP. That is the pattern described in the Optuna tutorial's "Multi-process optimization" section.


## Comment:

I am sorry, maybe my initial code was misleading. In fact, `main` run on different
ranks is supposed to sample different configurations and run different trials. I
missed sending different seeds to the devices, this was my mistake.

Also, please note I am using `DDPStrategy` only for convenience. I am NOT using it
to accumulate gradients between ranks:

* In `keys_values.autotune.autotune_full.eval_autotune_metrics`, the call
  `loss.backward()` runs locally on each rank, there is NO accumulation.
* In `keys_values.finetune.longcontext_full.validate`, the validation loss is
  computed locally on each rank, there is NO accumulation.

Could you please:

1. Fix my mistake, by sending different seeds to `main`, so the seed depends on the
   rank.
2. Fix your code as well. I do not want the different ranks to evaluate the same
   configurations. This would not make sense, because only rank 0 would do anything
   useful. I want them to run different trials, as overseen by Optuna. Optuna needs
   to send different trials to the different ranks, and it needs to receive results
   from each rank.

### Answer from Claude Code:

All tests pass, including the multi-worker test which confirms that 4 concurrent workers each got unique trial numbers (the gaps in the sequence are FAIL trials for invalid combinations, which Optuna counts internally but don't appear as evaluations).

Summary of changes:

`autotune_full.py` — two fixes:

* `fabric.seed_everything(seed + fabric.global_rank)` — each rank gets a distinct seed so they diverge in their random data sampling and don't duplicate work.
* Removed the `and fabric.global_rank == 0` guard from `tell_optuna_trial` — every rank runs its own trial and must report its own results.

`optuna_study.py` — ask_optuna_trial simplified:

* Removed the `fabric` parameter, the rank-0 guard, and the `dist.broadcast_object_list` broadcast entirely.
* Each rank calls `study.ask()` directly. The `JournalFileBackend` file lock ensures the two ask calls are serialised and return different trials — this is exactly the multi-process pattern from the Optuna tutorial, just using one process per rank rather than one process per machine.


## Comment:

Your solution creates an independent study per work, via
`study = create_optuna_study(optuna_config)` in `main`. This is not what I want. I'd
like Optuna to use multi-process optimization. A single tuner should provide candidates
to each worker and receive results from each worker. There should be single study
passed to each worker (i.e., call of `main`). You need to figure out (1) where this
study is run, and (2) how the workers communicate.

### Answer from Claude Code:

All tests pass. Here is a summary of the changes and the reasoning:

**The correct architecture:**


setup_internal (single process, before fabric.launch)
  └── create_optuna_study(config)   ← study record written to journal file once

`fabric.launch(main, ...) `           ← spawns N worker processes

*  rank 0: open_optuna_study(config) → ask → eval → tell
*  rank 1: open_optuna_study(config) → ask → eval → tell
*  rank N: open_optuna_study(config) → ask → eval → tell

`optuna_study.py` — split the old create_optuna_study into two functions:

* `create_optuna_study` — creates the study with load_if_exists=False. Called once in setup_internal. This is the single point where the study's sampler, directions, and metric names are configured.
* `open_optuna_study` — calls optuna.load_study, connecting to the pre-existing record. Called by each worker in main. A fresh sampler instance is constructed per worker (Optuna samplers are not process-safe to share).

`autotune_full.py` — two changes:

* `setup_internal`: calls create_optuna_study just before fabric.launch, so the study exists before any worker starts.
* `main`: calls open_optuna_study instead of create_optuna_study.


## Comment:

In `open_optuna_study`, a new sampler is created calling `_make_sampler`. Then, the
common study is loaded. This is still not what I want:

* Each sampler on a rank makes its decisions independently of each other. If
  `TPESampler` is used, I would expect that results from ALL ranks are collected
  in the same sampler, and then decisions for the different ranks are made in this
  one place.
* Say the sampler running on rank 0 has sampled a certain part of the space well.
  If the sampler on rank 1 does not know about this, it will sample suboptimally.

If Optuna is able to do multi-process optimization in this way at all, i.e. using
a SINGLE sampler than asks workers on several devices and receives results from
several ranks, then please do implement that.

### Answer from Claude Code:

The sampler object is stateless with respect to trial outcomes. Every call to sample_relative (the method that actually proposes a new configuration) reads completed trials from storage:

* `TPESampler`: calls `study._get_trials(...)` on every invocation — it builds its probability models fresh from the full trial history in the file each time.
* `NSGAIISampler`: calls `study._get_trials(...)` and `study._storage.get_study_system_attrs(...)` to reconstruct the population from completed trials stored in the backend.

* The separate sampler objects per rank hold only an RNG and configuration parameters. There is no in-process cache of trial outcomes that one rank would be missing from another. Optuna is deliberately designed this way to support exactly this multi-process pattern.

The one real issue was TPE without constant_liar: while the above is true for completed trials, a naive TPE run can propose similar points to two ranks that are both currently running (neither has called tell yet). With constant_liar=True (now the default), TPE writes the proposed parameters into storage immediately at ask time, so the other rank's sampler reads them as pseudo-observations and diverges. NSGA-II handles this implicitly through its population/generation bookkeeping, so it needed no change.


## Comment:

Now, I need some code and instructions for how to visualize the results. Use what Optuna
provides if possible. This is a constrained optimization problem with two objectives.
A good visualization would show the Pareto curve for all trials, with feasible and
infeasible trials marked differently.

### Answer from Claude Code:

New file: `keys_values/autotune/visualize_study.py`

Run it from the repo root after a tuning run:

```bash
~/virtenvs/keysvals_optuna/bin/python keys_values/autotune/visualize_study.py \
    --storage path/to/journal.log \
    --study   autotune \
    --out     results/
--no-browser suppresses auto-opening if you're on a headless machine. You also need plotly installed (pip install plotly).
```

Four HTML files are produced:

* `pareto_front.html` — the main plot. Scatter of (time_train, time_eval) for all completed trials, with three colour-coded series automatically determined from the constraints system attribute written by the sampler:
  - Pareto-front / Best (red scale): feasible and non-dominated.
  - Feasible (blue scale): feasible but dominated.
  - Infeasible (grey): OOM or runtime-exception trials.
* `hypervolume_history.html` — scalar summary of Pareto-front quality over time. The reference point is set to 110% of the worst observed feasible value on each axis, which is a standard conservative choice. Useful for comparing sampler configurations or checking convergence.
* `param_importances_time_train.html` and `param_importances_time_eval.html` — fANOVA-based importance for each parameter, one plot per objective. Tells you which variables actually drive time_train vs time_eval, so you can narrow the search space for future runs. Skipped if fewer than 4 feasible trials are available (fANOVA needs a minimum sample).
