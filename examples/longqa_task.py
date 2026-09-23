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
"""
Controlled long-context QA: key-value retrieval ("needle in a haystack").

Builds prompts of a target token length filled with ``The value for key
<key> is <value>.`` lines, one of which is the target. The model is asked for
the target key's value; the answer is verifiable by substring match. This is
the standard probe for whether a sparse KV cache (H2O) evicts the token the
task needs -- exactly the quality question for long-context sparse attention.

Shared by the quality-eval and GRPO-training scripts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from litgpt.tokenizer import Tokenizer


@dataclass
class QAExample:
    prompt_ids: torch.Tensor  # 1D token ids (chat-templated)
    target: str  # the value string to retrieve
    key: str
    needle_frac: float  # relative position of the needle in the filler


def _rand_key(rng: random.Random) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))


def _rand_value(rng: random.Random) -> str:
    return str(rng.randint(100000, 999999))


TASK_MODES = ("needle_last", "needle_first", "recency")


def _filler_line(rng: random.Random) -> str:
    return f"The value for key '{_rand_key(rng)}' is {_rand_value(rng)}."


def _build_records(tokenizer, needle_line, needle_frac, budget_tokens, rng) -> str:
    """Filler records with the needle inserted at ``needle_frac`` of the way
    through (frac >= 1.0 => appended last / most recent)."""
    lines: List[str] = []
    tok = 0
    placed = False
    while tok < budget_tokens:
        if needle_frac < 1.0 and not placed and tok >= needle_frac * budget_tokens:
            line, placed = needle_line, True
        else:
            line = _filler_line(rng)
        lines.append(line)
        tok += int(tokenizer.encode(" " + line).size(0))
    if needle_frac >= 1.0:
        lines.append(needle_line)  # needle is the most recent record
    elif not placed:
        lines.insert(len(lines) // 2, needle_line)
    return " ".join(lines)


def build_example(
    tokenizer: Tokenizer,
    apply_prompt_style,
    context_len: int,
    rng: random.Random,
    device=None,
    needle_frac: float | None = None,
    mode: str = "needle_last",
) -> QAExample:
    """Build one key-value QA example of ~``context_len`` tokens.

    All modes use the SAME working "Question ... Answer:" framing (so the dense
    baseline is high); only two things vary, to probe the sparse-attention
    frontier:

    - ``needle_last``:  question AFTER the records; needle at a random position.
      H2O's worst case -- during prefill it can't know which record matters, so
      the needle isn't a heavy hitter and gets evicted.
    - ``needle_first``: question BEFORE the records; needle at a random position.
      The key is known during prefill, so attention (and H2O's scoring) can land
      on the needle. Tests whether query-awareness rescues H2O.
    - ``recency``:      question after the records; needle is the LAST record.
      Recency-keeping policies (lastrec, H2O's recent window) should retain it.
    """
    assert mode in TASK_MODES, mode
    question_key = _rand_key(rng)
    target = _rand_value(rng)
    needle_line = f"The value for key '{question_key}' is {target}."
    question = (f"Question: what is the value for key '{question_key}'? "
                "Reply with only the value.")

    if mode == "recency":
        needle_frac = 1.0
    elif needle_frac is None:
        needle_frac = rng.uniform(0.15, 0.85)

    if mode == "needle_first":
        head = question + "\n\nRecords:\n"
        tail = "\n\nAnswer:"
    else:  # needle_last / recency: question comes after the records
        head = "Records:\n"
        tail = "\n\n" + question + "\nAnswer:"

    overhead = int(tokenizer.encode(head + tail).size(0))
    budget = max(context_len - overhead - 32, 32)
    records = _build_records(tokenizer, needle_line, needle_frac, budget, rng)
    body = head + records + tail

    ids = tokenizer.encode(apply_prompt_style(body), device=device)
    return QAExample(prompt_ids=ids, target=target, key=question_key,
                     needle_frac=needle_frac)


def build_dataset(
    tokenizer: Tokenizer,
    apply_prompt_style,
    context_len: int,
    n_examples: int,
    seed: int = 0,
    device=None,
    mode: str = "needle_last",
) -> List[QAExample]:
    rng = random.Random(seed)
    return [
        build_example(tokenizer, apply_prompt_style, context_len, rng,
                      device=device, mode=mode)
        for _ in range(n_examples)
    ]


def left_pad(sequences: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(int(s.size(0)) for s in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, max_len - seq.size(0):] = seq
    return out
