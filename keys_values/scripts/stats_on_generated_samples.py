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
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Optional, Any, Tuple
import yaml

from transformers import AutoTokenizer

from keys_values.scripts.cleanup_evaluation import datasets_and_cases

BASE_MODEL_PATH = "/home/ubuntu/git/keys_values/checkpoints/Qwen/Qwen3-4B-Instruct-2507"


def _length_ratio(
    output: str,
    target: str,
    tokenizer: Optional[Any],
) -> Tuple[float, int]:
    if tokenizer is None:
        return len(output) / len(target), len(output)
    else:
        encoded_output = tokenizer.encode(output, add_special_tokens=False)
        encoded_target = tokenizer.encode(target, add_special_tokens=False)
        return len(encoded_output) / len(encoded_target), len(encoded_output)


def _extract_number(
    record: dict,
    tokenizer: Optional[Any],
) -> Tuple[float, int]:
    output = record["output"]
    raw_target = record["raw_target"]
    if isinstance(raw_target, str):
        target = raw_target
    elif record.get("sub_exact_match") == 1.0:
        target = max(
            (t for t in raw_target if t in output),
            key=len,
            default=max(raw_target, key=len),
        )
    else:
        if "sub_exact_match" not in record:
            print("UUPS!")
        target = max(raw_target, key=len)
    return _length_ratio(output, target, tokenizer)


def _find_yaml_files(
    base_path: Path,
    dataset: str,
    case: str,
    eval_dir: str,
    search_for_setups: bool,
) -> List[Path]:
    if not search_for_setups:
        directory = base_path / dataset / case / eval_dir
        if directory.exists():
            return sorted(directory.glob("generated_samples_*.yaml"))
        else:
            return []
    # Search for setup subdirectory that contains eval_dir
    case_path = base_path / dataset / case
    if case_path.exists():
        for setup_dir in case_path.iterdir():
            if not setup_dir.is_dir():
                continue
            name = setup_dir.name
            if name == "final" or (name.startswith("step-") and name[5:].isdigit()):
                candidate = setup_dir / eval_dir
                if candidate.is_dir():
                    return sorted(candidate.glob("generated_samples_*.yaml"))
    return []


def _mean_and_std(entries: List[float]) -> Tuple[float, float]:
    num = len(entries)
    mean = sum(entries) / num
    std = math.sqrt(sum((x - mean) ** 2 for x in entries) / num)
    return mean, std


@dataclass
class Statistics:
    ratio: Tuple[float, float]
    frac_at_max_length: Optional[Tuple[float, float]]

    @staticmethod
    def from_data(
        ratios: List[float],
        fracs_at_max_length: Optional[List[float]] = None,
    ) -> "Statistics":
        return Statistics(
            ratio=_mean_and_std(ratios),
            frac_at_max_length=(
                None
                if fracs_at_max_length is None
                else _mean_and_std(fracs_at_max_length)
            ),
        )


def stats_for(
    dataset: str,
    case_key: str,
    base_path: Path,
    eval_dir: str,
    search_for_setups: bool,
    tokenizer: Optional[Any],
    max_tokens: int,
    verbose: bool,
) -> Optional[Statistics]:
    yaml_files = _find_yaml_files(
        base_path, dataset, case_key, eval_dir, search_for_setups
    )
    number_pairs: List[Tuple[float, int]] = []
    for path in yaml_files:
        with open(path) as f:
            records = yaml.safe_load(f)
        for record in records:
            number_pairs.append(_extract_number(record, tokenizer))
    num_entries = len(number_pairs)
    if num_entries == 0:
        return None
    ratios = [x[0] for x in number_pairs]
    if tokenizer is not None:
        num_tokens = [x[1] for x in number_pairs]
        fracs_at_max_length = [float(x >= max_tokens) for x in num_tokens]
        if verbose:
            num_eq_max = sum(x == max_tokens for x in num_tokens)
            num_gt_max = sum(x > max_tokens for x in num_tokens)
            hist_lt_max = Counter([x for x in num_tokens if x < max_tokens])
            print(
                f"  ({dataset}, {case_key}): num_eq_max={num_eq_max}, num_gt_max={num_gt_max}, vals_lt_max={dict(hist_lt_max)}"
            )
    else:
        fracs_at_max_length = None
    return Statistics.from_data(ratios, fracs_at_max_length)


def _wrap_values(
    vals: Tuple[float, float],
    is_ratio: bool,
) -> str:
    patterns = ["{x:.1f}"] * 2
    if is_ratio:
        multipliers = [1] * 2
    else:
        multipliers = [100] * 2
    parts = [p.format(x=x * m) for p, x, m in zip(patterns, vals, multipliers)]
    return r"{\small\!" + parts[0] + r"$\pm$\!" + parts[1] + r"}"


def _col_header(
    col_labels: List[str],
    do_tokenize: bool,
) -> List[str]:
    num_cols = len(col_labels)
    if do_tokenize:
        col_spec = "|l|l|" + ("rr|" * num_cols)
        col_strings = [r"\multicolumn{2}{c|}{" + x + r"}" for x in col_labels]
        tex_lines = [
            r"\begin{tabular}{" + col_spec + "}",
            r"\hline",
            " & ".join(["", "trn"] + col_strings) + r" \\",
            " &" + (" & R & p" * num_cols) + r" \\",
            r"\hline\hline",
        ]
    else:
        col_spec = "|l|l|" + ("r" * num_cols) + "|"
        tex_lines = [
            r"\begin{tabular}{" + col_spec + "}",
            r"\hline",
            " & ".join(["", "trn"] + col_labels) + r" \\",
            r"\hline\hline",
        ]
    return tex_lines


def _row_for_stats(tex_lines: List[str], stats: List[Statistics]):
    for i, stat in enumerate(stats):
        tail = r" \\" if i == len(stats) - 1 else r" &"
        if stat is not None:
            row = "  " + _wrap_values(stat.ratio, is_ratio=True)
            if do_tokenize:
                row += " & " + _wrap_values(stat.frac_at_max_length, is_ratio=False)
        else:
            row = "  -"
            if do_tokenize:
                row += " & -"
        row += tail
        tex_lines.append(row)


# Table has datasets as columns, cases as rows. This works for few datasets,
# many cases.
def main_type1(
    datasets: List[str],
    cases: List[Tuple[str, str]],
    result_path: Path,
    eval_dir: str,
    search_for_setups: bool,
    tokenizer: Optional[Any],
    max_tokens: int,
    verbose: bool,
    tag: str,
):
    do_tokenize = tokenizer is not None
    base_path = result_path.parent
    col_labels = [
        d.removeprefix("helmet_").rsplit("_", 1)[0].replace("_", r"\_")
        for d in datasets
    ]
    tex_lines = _col_header(col_labels, do_tokenize)

    for case_key, case_label in cases:
        case_label = case_label.replace("_", r"\_")
        stats = [
            stats_for(
                dataset=dataset,
                case_key=case_key,
                base_path=base_path,
                eval_dir=eval_dir,
                search_for_setups=search_for_setups,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                verbose=verbose,
            )
            for dataset in datasets
        ]
        tex_lines.append(r"\rule{0pt}{13pt} " + case_label + r" &")
        _row_for_stats(tex_lines, stats)

    tex_lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
        ]
    )
    if result_path.exists():
        result_path.unlink()
    print(f"Writing result table to {result_path}")
    result_path.write_text("\n".join(tex_lines) + "\n")


# Table has datasets as rows, cases as columns. This works for many datasets,
# few cases.
def main_type2(
    datasets: List[str],
    cases: List[Tuple[str, str]],
    result_path: Path,
    eval_dir: str,
    search_for_setups: bool,
    tokenizer: Optional[Any],
    max_tokens: int,
    verbose: bool,
    tag: str,
):
    do_tokenize = tokenizer is not None
    base_path = result_path.parent
    row_labels = [
        d.removeprefix("helmet_").rsplit("_", 1)[0].replace("_", r"\_")
        for d in datasets
    ]
    col_labels = [c[1].replace("_", r"\_") for c in cases]
    tex_lines = _col_header(col_labels, do_tokenize)

    for dataset, row_label in zip(datasets, row_labels):
        stats = [
            stats_for(
                dataset=dataset,
                case_key=case[0],
                base_path=base_path,
                eval_dir=eval_dir,
                search_for_setups=search_for_setups,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                verbose=verbose,
            )
            for case in cases
        ]
        tex_lines.append(r"\rule{0pt}{11pt} " + row_label + r" & " + tag + " &")
        _row_for_stats(tex_lines, stats)

    tex_lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
        ]
    )
    if result_path.exists():
        result_path.unlink()
    print(f"Writing result table to {result_path}")
    result_path.write_text("\n".join(tex_lines) + "\n")


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"
    do_tokenize = True
    verbose = True
    table_type_1 = False

    max_tokens = 128
    eval_dir = "eval_128"
    # dataset_size = "64k"
    dataset_size = "128k"
    # is_rerun = False
    is_rerun = True
    is_baseline = False
    # is_baseline = True
    is_base_model = False
    # is_base_model = True
    extra_data = False
    # extra_data = True
    if is_rerun:
        base_path = base_path / "rerun"
    elif is_baseline:
        base_path = base_path / "baseline"
    elif is_base_model:
        base_path = base_path / "basemod"
    if is_baseline:
        tag = "sp"
    elif is_base_model:
        tag = "no"
    else:
        tag = "us"
    search_for_setups = not is_baseline and not is_base_model
    datasets, cases = datasets_and_cases(
        dataset_size,
        extra_data,
        is_baseline,
        is_base_model,
        with_short=True,
    )
    if not table_type_1 and not extra_data:
        print("Table type 2: Reducing to 3 cases")
        cases = [
            ("slr_4gpu_cs1024_lr5", "slr_1024"),
            ("h2onorm_4gpu_cs1024_lr5", "h2onorm_1024"),
            ("h2oorig_4gpu_cs1024_lr5", "h2oorig_1024"),
        ]

    if do_tokenize:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    else:
        tokenizer = None
    result_path = base_path / f"stats_samples_{dataset_size}.tex"

    kwargs = dict(
        datasets=datasets,
        cases=cases,
        result_path=result_path,
        eval_dir=eval_dir,
        search_for_setups=search_for_setups,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        verbose=verbose,
        tag=tag,
    )
    if table_type_1:
        main_type1(**kwargs)
    else:
        main_type2(**kwargs)
