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
import re
from difflib import SequenceMatcher
from collections import Counter
import math
from decimal import Decimal, InvalidOperation
from typing import Iterable, Optional, List, Dict, Set

from .response_parser import normalize_string_response


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _normalize_for_tokens(s: str) -> str:
    s = (s or "").strip().lower()
    # keep letters+digits, turn everything else into spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_multiset_similarity(a_tokens, b_tokens) -> float:
    """
    Multiset (bag-of-words) Jaccard: |A∩B| / |A∪B| counting duplicates.
    """
    ca, cb = Counter(a_tokens), Counter(b_tokens)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return 0.0 if union == 0 else inter / union


def sub_exact_similarity(a: str, b: str) -> float:
    """
    Combines:
      - order-insensitive token multiset Jaccard
      - character similarity on sorted tokens (handles minor typos)

    Returns a similarity score in [0, 1].
    """
    na, nb = _normalize_for_tokens(a), _normalize_for_tokens(b)

    a_tokens = na.split() if na else []
    b_tokens = nb.split() if nb else []

    token_score = _token_multiset_similarity(a_tokens, b_tokens)

    # backstop for small variations/typos while remaining mostly order-insensitive
    a_sorted = " ".join(sorted(a_tokens))
    b_sorted = " ".join(sorted(b_tokens))
    char_score = (
        SequenceMatcher(None, a_sorted, b_sorted).ratio()
        if (a_sorted or b_sorted)
        else 0.0
    )

    return max(token_score, char_score)


def sub_exact_match(
    response: str, target_value: str | int | float, *, threshold: float = 0.90
) -> bool:
    """
    Determine whether the response and target_value have substring exact match (normalized).
    """
    response = normalize_string_response(response)
    target_value = str(target_value)
    if response in target_value or target_value in response:
        is_match = True
    else:
        match_score = sub_exact_similarity(response, target_value)
        is_match = match_score >= threshold
    return is_match


def exact_match(response: str, target_value: str | int | float) -> bool:
    """
    Determine whether the response and target_value have exact match (normalized).
    """
    response = normalize_string_response(response)
    target_value = str(target_value)
    return response == target_value


def ndcg_at_10_ranked_numbers(
    pred: Iterable[str], target: Iterable[str], *, k: int = 10
) -> float:
    """
    NDCG@k for ranked numeric string lists.
    - Target list is the ideal ranking.
    - Relevance is derived from target rank: rel = (k - idx) for idx < k, else 0.
    - Prediction duplicates are only credited once.

    Returns a float in [0, 1].
    """

    def canon_num(x: str) -> Optional[str]:
        if x is None:
            return None
        t = str(x).strip()
        if not t:
            return None
        try:
            d = Decimal(t)
        except (InvalidOperation, ValueError):
            return None
        if d == 0:
            d = Decimal(0)  # avoid "-0"
        # normalized decimal -> plain string without trailing zeros
        s = format(d.normalize(), "f").rstrip("0").rstrip(".")
        return s if s else "0"

    def dcg(rels: List[float]) -> float:
        total = 0.0
        for i, rel in enumerate(rels):
            total += (2.0**rel - 1.0) / math.log2(i + 2)
        return total

    # Canonicalize + keep order
    pred_c = [canon_num(x) for x in pred]
    pred_c = [x for x in pred_c if x is not None][:k]

    target_c_raw = [canon_num(x) for x in target]
    target_c_raw = [x for x in target_c_raw if x is not None]

    # Build unique target list in order (ideal ranking items)
    target_unique: List[str] = []
    seen_t: Set[str] = set()
    for x in target_c_raw:
        if x not in seen_t:
            seen_t.add(x)
            target_unique.append(x)

    if not target_unique or k <= 0:
        return 0.0

    # Relevance map from target rank (only top-k have >0 relevance)
    rel_map: Dict[str, float] = {}
    for idx, val in enumerate(target_unique[:k]):
        rel_map[val] = float(k - idx)

    # Predicted relevances (credit each target value at most once)
    used: Set[str] = set()
    pred_rels: List[float] = []
    for x in pred_c:
        if x in rel_map and x not in used:
            pred_rels.append(rel_map[x])
            used.add(x)
        else:
            pred_rels.append(0.0)

    dcg_val = dcg(pred_rels)

    # Ideal DCG: perfect ordering of the top-k target items
    ideal_rels = [float(k - idx) for idx in range(min(k, len(target_unique)))]
    idcg_val = dcg(ideal_rels)

    return 0.0 if idcg_val == 0.0 else (dcg_val / idcg_val)


def rouge_n_f1(response: str, target: str, *, n: int = 1) -> float:
    """
    Compute ROUGE-N precision/recall/F1 between `response` and `target`.

    Returns a float in [0, 1].
    """
    response = normalize_string_response(response) or ""
    resp_norm = _normalize_for_tokens(response)
    targ_norm = _normalize_for_tokens(target)

    resp_tokens = resp_norm.split() if resp_norm else []
    targ_tokens = targ_norm.split() if targ_norm else []

    resp_ng = _ngram_counts(resp_tokens, n)
    targ_ng = _ngram_counts(targ_tokens, n)

    resp_total = sum(resp_ng.values())
    targ_total = sum(targ_ng.values())

    if resp_total == 0 and targ_total == 0:
        precision, recall, f1 = (1.0, 1.0, 1.0)
        return 1.0
    if resp_total == 0 or targ_total == 0:
        precision, recall, f1 = 0.0, 0.0, 0.0
        return 0.0

    overlap = sum((resp_ng & targ_ng).values())

    precision = overlap / resp_total
    recall = overlap / targ_total
    f1 = (
        0.0
        if (precision + recall) == 0
        else (2 * precision * recall) / (precision + recall)
    )
    return f1
