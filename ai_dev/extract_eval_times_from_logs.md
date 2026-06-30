# Write `extract_eval_times_from_logs.py` script

## Initial prompt

Please write a Python script which extracts time values from log files, stores
them into a result CSV file, and creates a table with statistics in LaTeX.

* Extend the code `keys_values/scripts/extract_eval_times_from_logs.py` by the
  `main` function.
* `main` should read log files `log_dir / f"gpu{i}.log"`, `i in range(4)`.
  It should extract tuples of values `(dataset, policy, rank, subdir, id1, id2, time_secs)`.
* Look for pairs of rows of the form:
  "[rank 0 | 2026-06-14 21:36:21]: Batch [406, 407]: sub_exact_match = 0.000, eval_time = 85325.27 ms"
  "Storing to /home/ubuntu/out/finetune/neurips_exp/lora/qwen3_4b/baseline/helmet_nq_128k/h2onorm_4gpu_cs1024_lr5/eval/eval_metrics_406.csv"
  From the filename, extract:
  - "/home/ubuntu/out/finetune/neurips_exp/lora/qwen3_4b" is fixed
  - `subdir`: "baseline". Can also be "basemod", or can be missing (use empty string for that).
    This is the same value for all rows.
  - `dataset`: "helmet_nq_128k". Starts with  "helmet_" or "longbench_"
  - `policy`: "h2onorm_4gpu_cs1024_lr5"
  - `f"eval/eval_metrics_{j}.csv"`: Nothing to extract
  From the row above, extract:
  - `rank`: 0 (also `i` from log file name `f"gpu{i}.log"`)
  - `id1`, `id2`: `(406, 407)` above, the list after "Batch ". Also of size 2
  - `time_secs`: Value after "eval_time = " divided by 1000
* Create a result CSV file, written to `log_dir / f"times_eval_{subdir}.csv"`,
  or `log_dir / f"times_eval.csv"` if `subdir` is missing,
  containing columns "dataset", "policy", "rank", "subdir", "id1", "id2", "time_secs".
* Apart from the CSV file, `main` should also write LaTeX code for a table, with
  columns for `dataset`, rows for `policy`. Cells should be
  `f"{mean:.2f} ({std:.2f})"`, where mean and std are over over all
  `(rank, id1, id2)` values. If a combination `(dataset, policy)` is not present,
  write "-".
* The LaTeX table should be formatted in the same way as in the script
  `keys_values/scripts/extract_train_times_from_logs.py` (copy code from there
  if possible). It should be written to
  `log_dir / f"times_eval_{subdir}.tex"`, or `log_dir / f"times_eval.tex"` if
  `subdir` is missing.

## Answer

I forgot something, could you please modify the script?

* In the case when `subdir` is missing, there is another field `task`. In this case,
  look for pairs of rows of the form:
  "[rank 0 | 2026-06-02 06:06:35]: Batch step-000440, [593, 592]: sub_exact_match = 0.000, eval_time = 43252.14 ms"
  "Storing to /home/ubuntu/out/finetune/neurips_exp/lora/qwen3_4b/helmet_nq_128k/slr_4gpu_cs2048_lr5/step-000440/eval/eval_metrics_593.csv"
  From the filename:
  - `subdir` is missing
  - `task`: "step-000440", comes after `policy`, before `f"eval/eval_metrics_{j}.csv"`.
    Values are either of the form `f"step-{no}"`, `no` a 6-digit number, or "final"
* In this case, extract `(dataset, policy, task, rank, id1, id2, time_secs)`, and
  write the CSV file with these columns.
* For the LaTeX table, if for some `(dataset, policy)` tuple, there are
  values `(task, rank, id1, id2, time_secs)` with several different `task` values,
  compute `mean` and `std` only on values for which `task == task_max`. Here,
  `task_max` is of the form `f"step-{no}"`, where `no` is maximum. If all `task`
  values are "final", then `task_max = "final"` as well.
* If `subdir == "baseline"` or `subdir == "basemod"`, the script should behave as before.
