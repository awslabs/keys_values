# Write `extract_train_times_from_logs.py` script

## Initial prompt

Please write a Python script which extracts time values from log files, stores
them into a result CSV file, and creates a table with statistics in LaTeX.

* Extend the code `keys_values/scripts/extract_train_times_from_logs.py` by the
  `main` function. `main` should iterate over all combinations
  `dataset in datasets, policy in policies`. For each, define
  `base_dir = base_path / dataset / policy` and `log_dir = base_dir / "logs"`.
  If for a combination `(dataset, policy)`, `base_dir` does not exist, skip it.
* The log file to extract from is either `f"{log_dir}/gpu0.log"`, or
  `f{log_dir}/resume{i}/gpu0.log"`, where `i` is a number. If several such
  files exist, only use the "resume" one with the largest `i`, do not use the
  others.
* Depending on `mode`, extract "train" or "valid" time values
* Create a result CSV file, written to `base_path / f"times_{mode}.csv"`,
  containing columns "dataset", "policy", "epoch", "iter", "time_secs".
* For `mode == "train"`, look for lines of the form
  "Epoch 0 | iter   1 | loss train: 24.250, val_loss valid: 22.348 | iter time: 231.637 s".
  Here, spacing may be different, and "val_loss" may be another string. Extract
  values for:
  - "epoch": Value after "Epoch" (0 above)
  - "iter": Value after "iter" (1 above)
  - "time_secs": Value after "iter time:" (231.637 above). For some logs, the unit
    may be "ms", then divide by 1000.
  Only use lines for which the "epoch" number is such that `filter_epochs(epoch)`
  returns `True`.
* For `mode == "valid"`, look for lines of the form
  "Epoch 0 | iter  10          | val_loss: 21.710 | val_time: 248.608 s".
  Here, spacing may be different, and "val_loss" may be another string. Extract
  values for:
  - "epoch": Value after "Epoch" (0 above)
  - "iter": Value after "iter" (10 above)
  - "time_secs": Value after "val_time:" (248.608 above). For some logs, the unit
    may be "ms", then divide by 1000.
  Only use lines for which the "epoch" number is such that `filter_epochs(epoch)`
  returns `True`.
* Apart from the CSV file, `main` should also write LaTeX code for a table, with
  columns for `dataset`, rows for `policy`. Cells should be
  `f"{mean:.2f} ({std:.2f})"`, where mean and std are over "epoch" and "iter".
  If a combination `(dataset, policy)` is skipped, write "-". Write the LaTeX
  code to `base_path / f"times_{mode}.tex"`.
