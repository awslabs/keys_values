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
from typing import Union, List, Optional


def normalize_string_response(
    text: str,
    *,
    prefixes: tuple[str, ...] = (
        "answer",
        "document",
        "response",
        "output",
        "label",
        "</think>",
    ),
) -> str:
    """
    Normalize a raw string response:
    - strip whitespace
    - remove leading prefixes like "Answer: " / "Document: " (case-insensitive)
    - remove leading clauses like "... the answer is: " / "... the label is "
    - unwrap one layer of surrounding quotes/backticks
    """
    if text is None:
        return ""

    s = str(text).strip()
    if not s:
        return ""

    # 1) Remove stacked simple prefixes at the very beginning: "Answer:", "Label -", "</think>", etc.
    prefix_alt = "|".join(re.escape(p) for p in prefixes if p)
    if prefix_alt:
        prefix_re = re.compile(
            rf"^(?:{prefix_alt})\s*(?:[:=\-]\s*|\s+)?", re.IGNORECASE
        )
        while True:
            new_s = prefix_re.sub("", s, count=1).lstrip()
            if new_s == s:
                break
            s = new_s

    # 2) Remove leading "lead-in clause" that introduces the answer/label/etc.
    #    Examples removed:
    #      "bla bla, the answer is: "   -> keep what's after
    #      "In summary the label is "   -> keep what's after
    #
    # We only search this cue within the first N chars to avoid nuking mid-body content.
    N = 240
    head = s[:N]

    fields_alt = "|".join(
        re.escape(p) for p in ("answer", "label", "document", "response", "output")
    )
    cue_re = re.compile(
        rf"""
        (?is)                           # ignorecase + dotall (for messy outputs)
        ^.*?                            # any leading text (non-greedy)
        \b(?:the\s+)?(?:{fields_alt})\b # "the answer"/"label"/...
        \s*
        (?:is\b\s*)?                    # optional "is"
        (?:
            [:=\-]\s*                   # ":", "=", "-"
            |                           # or
            \s+                         # just whitespace (for "answer is 42")
        )
        """,
        re.VERBOSE,
    )

    m = cue_re.search(head)
    if m:
        s = s[m.end() :].lstrip()

    # 3) Unwrap one layer of surrounding quoting/backticks, e.g. "`foo`", '"foo"', "'foo'"
    if len(s) >= 2 and ((s[0], s[-1]) in {("`", "`"), ('"', '"'), ("'", "'")}):
        s = s[1:-1].strip()

    return s


def extract_numbers(text: str, *, as_list: bool = True) -> Union[List[str], str]:
    """
    Extract numeric values from free-form text.

    - If as_list=True: returns a list of numbers in string. If none found, returns ["0"].
    - If as_list=False: returns the first found number in string. If none found, returns "0".
    """
    # extract float numbers can provide more coverage than integer numbers
    number_re = re.compile(
        r"""
            (?<![\w.])                 # don't start in the middle of a word/identifier
            [-+]?                      # optional sign
            (?:
                (?:\d{1,3}(?:,\d{3})+) # 1,234 style
                |                      # or
                \d+                    # plain digits
            )
            (?:\.\d+)?                 # optional decimal part
            (?:[eE][-+]?\d+)?          # optional scientific notation
            (?:
                \.(?=\s*(?!\d))   # allow trailing '.' only if next non-space isn't a digit
            )?
            (?![\w.])                  # don't end in the middle of a word/identifier
            """,
        re.VERBOSE,
    )
    if not text:
        return ["0"] if as_list else "0"

    matches = number_re.findall(text)

    nums: List[float] = []
    for m in matches:
        # remove thousands separators like 1,234 -> 1234
        cleaned = m.replace(",", "")
        try:
            nums.append(float(cleaned))
        except ValueError:
            # regex should prevent most bad parses
            continue

    str_nums = [str(int(num)) for num in nums]
    if as_list:
        return str_nums if str_nums else ["0"]
    return str_nums[0] if str_nums else "0"


def extract_choice_letter(
    text: str,
    *,
    choices: str = "ABCD",
    default: str = "",
) -> str:
    """
    Extract a single multiple-choice letter (e.g., A/B/C/D) from free-form text.

    - Matches standalone letters, including forms like: "C", "C.", "C)", "(C)", "Answer: C", "option D"
    - If multiple candidates exist, prefers ones near cue words (answer/option/choice/select/correct).
    - Returns `default` if nothing is found.

    choices: string of allowed letters, e.g. "ABCDE" or "ABCD".
    """
    if not text:
        return default

    allowed = set(c.upper() for c in choices if c.strip())
    if not allowed:
        return default

    # Standalone letter token with optional surrounding punctuation, but NOT inside words.
    # Examples matched: "C", "C.", "C)", "(C)", "[C]", "Answer: C", "option D"
    token_re = re.compile(
        r"""
        (?<![A-Za-z0-9])          # not preceded by alnum (avoid inside words/ids)
        [\(\[\{\'"]*              # optional left punctuation
        (?P<ch>[A-Za-z])          # the letter
        [\)\]\}\'"]*              # optional right punctuation
        (?:
            [\.\):,\-]?           # optional trailing punctuation like ".", ")", ":", ",", "-"
        )?
        (?![A-Za-z0-9])           # not followed by alnum (avoid inside words)
        """,
        re.VERBOSE,
    )

    cue_re = re.compile(
        r"\b(answer|correct|option|choice|select|selected|pick)\b",
        re.IGNORECASE,
    )

    candidates = []
    for m in token_re.finditer(text):
        ch = m.group("ch").upper()
        if ch not in allowed:
            continue

        # Score by proximity to cue words in a small window around the match
        start, end = m.span()
        window_start = max(0, start - 40)
        window_end = min(len(text), end + 40)
        window = text[window_start:window_end]

        score = 0
        if cue_re.search(window):
            score += 10

        # Extra bonus if patterns like "answer: X" appear tightly
        tight = text[max(0, start - 15) : min(len(text), end + 15)]
        if re.search(
            r"(answer|correct)\s*[:\-]?\s*$",
            tight[: (start - max(0, start - 15))],
            re.IGNORECASE,
        ):
            score += 5

        candidates.append((score, start, ch))

    if not candidates:
        return default

    # Pick best by score, then by later position (often final answer is later)
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[-1][2]


def extract_value_token(text: str, *, default: str = "") -> str:
    """
    Extract the most likely 'answer value' token from an LLM output.

    Prefers tokens after cue phrases like "is:" / "are:".
    Within the candidate region, prefers:
      1) hyphenated hex tokens (UUID-ish, variable hyphen count)
      2) integers

    Returns `default` if nothing is found.
    """
    if not text:
        return default

    hyphen_hex_re = re.compile(r"\b[0-9a-fA-F]{4,}(?:-[0-9a-fA-F]{1,})+\b")
    int_re = re.compile(r"\b\d+\b")
    cue_re = re.compile(r"\b(?:is|are)\b\s*:?\s*", re.IGNORECASE)

    def clean_leading(s: str) -> str:
        s = s.lstrip(" \t\r\n:=-")
        s = re.sub(r"^(?:\*\*|__)+", "", s)  # markdown bold markers
        return s.lstrip()

    def pick_from_chunk(chunk: str) -> Optional[str]:
        # Prefer a backticked token early (often the answer)
        m_bt = re.search(r"`([^`]+)`", chunk)
        if m_bt:
            inside = m_bt.group(1).strip()
            m = hyphen_hex_re.search(inside)
            if m:
                return m.group(0)
            m = int_re.search(inside)
            if m:
                return m.group(0)

        m = hyphen_hex_re.search(chunk)
        if m:
            return m.group(0)

        m = int_re.search(chunk)
        if m:
            return m.group(0)

        return None

    # 1) Prefer extraction after "is/are(:)"
    for m in cue_re.finditer(text):
        tail = clean_leading(text[m.end() : m.end() + 240])
        val = pick_from_chunk(tail)
        if val:
            return val

    # 2) Fallback: last UUID-ish token, else last integer
    all_uuid = hyphen_hex_re.findall(text)
    if all_uuid:
        return all_uuid[-1]

    all_int = int_re.findall(text)
    if all_int:
        return all_int[-1]

    return default
