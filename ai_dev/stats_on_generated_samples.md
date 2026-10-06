# Write `stats_on_generated_samples.py` script

## Initial prompt

I need a Python script which computes certain statistics from results of an
experiment.

* I started a script in `~/git/keys_values/keys_values/scripts/stats_on_generated_samples.py`.
  Extend this file by adding the `main` function.
* Statistics are computed from files `generated_samples_176.yaml`. If
  `search_for_setups == False`, these are in
  `base_path / dataset / case / eval_dir`, where `case in cases`.
* If `search_for_setups == True`, they are in `base_path / dataset / case / setup / eval_dir`,
  where `setup` is of the form `f"step-{06d}"` or "final". Only one such path
  `base_path / dataset / case / setup` should have a subdirectory `eval_dir`.
* Each file `generated_samples_*.yaml` contains a list of 1 or 2 records, each
  with fields "idx", "raw_target", "output", " The value `record["output"]` is a
  string, the value `record["raw_target"]` is a string or a list of strings.
* If `record["raw_target"]` is a string, the number to extract per record is
  `len(record["output"]) / len(record["raw_target"])`.
* If `record["raw_target"]` is a list of strings, there is also a field
  `record["sub_exact_match"]` in each record, with values 0.0 or 1.0. If
  `record["sub_exact_match"] == 1.0`, one of the entries of `record["raw_target"]`
  is a substring of `record["output"]`. If this entry is called `target`, the
  number of extract is `len(record["output"]) / len(target)`.
  If `record["sub_exact_match"] == 1.0`, choose the longest entry from
  `record["raw_target"]` as value for `target`.
* The script should output mean and standard deviation of these numbers, for
  each `(dataset, case)` value (so over result files and records). It should
  also output mean and standard deviation over cases, result files, and
  records.
