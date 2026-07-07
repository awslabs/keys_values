# Write `extract_train_times_from_logs.py` script

## Initial prompt

Please write a Python script which extracts validation loss values from log
files and computes certain results.

* Extend the code `keys_values/scripts/extract_valid_loss_from_logs.py` by the
  `main` function. `main` should parse log files of names
  `log_path / "gpu0.log"` or `log_path / f"resume*/gpu0.log"`, where `*` is
  some positive integer. If no such files exist, print
  `f"{dataset}, {policy}: No logs"`.
* Otherwise, look for lines of the form
  "Epoch 0 | iter  10          | val_loss: 23.362 | val_time: 596.572 s"
  or
  "Initial evaluation          | val_loss: 23.774 | val_time: 595.956 s"
  across all log files. Extract values for:
  - `iter`: Value after "iter ". For "Initial evaluation", use `iter = 0`
  - `val_loss`: Value after "val_loss"
* From these, determine values `(iter_0, val_loss_0)`, where `val_loss_0`
  is the minimum value over all `val_loss`.
* Also, determine `(iter_1, val_loss_1)` as follows. Say that lists `iter`,
  `val_loss` are lists of extracted values, sorted by increasing `iter`.
  For `i > 0`, define `improve[i] = all(val_loss[i] < x for x in val_loss[:i])`,
  then `cnt[0] = 0`, and for any `i > 0`:
  `cnt[i] = 0 if improve[i] else cnt[i - 1] + 1`.
  Finally, `i_star = min(i for i, v in enumerate(cnt) if v >= patience)`.
  If the sequence in `min(...)` is empty, then `i_star = len(iter)`.
  Find a solution which is linear in `len(iter)`, not quadratic.
  Then: `(iter_1, val_loss_1)` is such that `val_loss_1` is the minimum
  of `val_loss[:i_star]`
* The code should output
  `f"{dataset}, {policy}: 0: step-{iter_0:06d} ({val_loss_0:.2f}) | 1: step-{iter_1:06d} ({val_loss_1:.2f})"`
