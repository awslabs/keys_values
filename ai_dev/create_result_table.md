# Write `create_result_table.py` script

## Initial prompt

I need a Python script which collects evaluation results for a number of datasets
and cases, computes averages, and write LaTeX code for a table containing these
average results.

* I started a script in `~/git/keys_values/keys_values/scripts/create_result_table.py`.
  Extend this file by adding the `main` function.
* For each `(dataset, case), dataset in datasets,
  case in [x[0] for x in cases]`, there is a CSV
  file `base_path / dataset / case / EVAL_METRICS_ALL_FILENAME`.
  The result file has columns `idx, task, sub_exact_match`. The `sub_exact_match`
  column contains metric values: average them over different `idx` for the same
  `task`.
* The result table has one column for each `dataset`, one row for each `case`.
* A cell can have several rows, one for each distinct `task` value contained in
  the result file for `(dataset, case)`. These `task` values can be different for
  every cell.
* The rows in a cell for `(dataset, case)` should be
  f"{short_task}:{avg_sub_exact_match:.4f}", where `avg_sub_exact_match` is the
  average `sub_exact_match` value for some `task`, and `short_task = "fin"` if
  `task == "final"`, `short_task = "120"` if `task == "step-000120"` (the last 3
  digits).
* If there is no result file for a combination `(dataset, case)`, the corresponding
  cell is left empty.
* Label the columns with the `dataset` name, stripping off "helmet_"
* Label the rows with `x[1]` for the tuple `x` in `cases`.
* The `main` function should write LaTeX code for the table into the file
  `result_path`.

### Answer

Done. The main function:

* Derives base_path from result_path.parent
* For each `(dataset, case)` pair, reads the CSV if it exists, groups by task, averages sub_exact_match, and formats each row as short_task:avg
* Maps task names via _short_task: "final" → "fin", others → last 3 digits
* Uses `\makecell{...}` for cells with multiple task rows (requires \usepackage{makecell} in the LaTeX preamble)
* Writes the full `tabular` environment to `result_path`


## Comment

The column and row labels may contain "_", which you need to change to "\_" in LaTeX. Could you please fix this?

### Answer

Fixed — underscores in both column and row labels are now escaped as \_.