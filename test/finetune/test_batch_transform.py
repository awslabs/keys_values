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
import pytest
import torch

from keys_values.data import INPUT_IDS_NAME, LABELS_NAME
from keys_values.finetune.batch_transform import SFTBatchTransform

PAD_ID = 0

IGNORE_INDEX = -3

EOS_ID = -1


def args_sft_batch_transform():
    return [
        (
            dict(
                input_ids=torch.tensor(
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, EOS_ID],
                        [10, 11, 12, 13, 14, 15, 16, 17, EOS_ID, PAD_ID],
                    ]
                ),
                labels=torch.tensor(
                    [
                        [
                            IGNORE_INDEX,
                            IGNORE_INDEX,
                            IGNORE_INDEX,
                            IGNORE_INDEX,
                            IGNORE_INDEX,
                            IGNORE_INDEX,
                            IGNORE_INDEX,
                            8,
                            9,
                            EOS_ID,
                        ],
                        [
                            IGNORE_INDEX,
                            IGNORE_INDEX,
                            IGNORE_INDEX,
                            IGNORE_INDEX,
                            14,
                            15,
                            16,
                            17,
                            EOS_ID,
                            IGNORE_INDEX,
                        ],
                    ]
                ),
            ),
            dict(
                input_ids=torch.tensor(
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, PAD_ID, PAD_ID],
                        [PAD_ID, PAD_ID, PAD_ID, 10, 11, 12, 13, 14, 15, 16, 17],
                    ]
                ),
                targets=torch.tensor(
                    [
                        [8, 9, EOS_ID, IGNORE_INDEX, IGNORE_INDEX],
                        [14, 15, 16, 17, EOS_ID],
                    ]
                ),
            ),
        ),
        (
            dict(
                input_ids=torch.tensor(
                    [
                        ([1] * 10) + [EOS_ID] + ([PAD_ID] * 6),
                        ([2] * 6) + [EOS_ID] + ([PAD_ID] * 10),
                        ([3] * 14) + [EOS_ID] + ([PAD_ID] * 2),
                        [4, 4, EOS_ID] + ([PAD_ID] * 14),
                        [5] * 16 + [EOS_ID],
                        ([6] * 8) + [EOS_ID] + ([PAD_ID] * 8),
                        ([7] * 5) + [EOS_ID] + ([PAD_ID] * 11),
                        ([8] * 13) + [EOS_ID] + ([PAD_ID] * 3),
                    ]
                ),
                labels=torch.tensor(
                    [
                        ([IGNORE_INDEX] * 8) + [1, 1, EOS_ID] + ([IGNORE_INDEX] * 6),
                        ([IGNORE_INDEX] * 5) + [2, EOS_ID] + ([IGNORE_INDEX] * 10),
                        ([IGNORE_INDEX] * 4)
                        + ([3] * 10)
                        + [EOS_ID]
                        + ([IGNORE_INDEX] * 2),
                        [IGNORE_INDEX, 4, EOS_ID] + ([IGNORE_INDEX] * 14),
                        ([IGNORE_INDEX] * 13) + ([5] * 3) + [EOS_ID],
                        [IGNORE_INDEX] + ([6] * 7) + [EOS_ID] + ([IGNORE_INDEX] * 8),
                        ([IGNORE_INDEX] * 4) + [7, EOS_ID] + ([IGNORE_INDEX] * 11),
                        ([IGNORE_INDEX] * 6)
                        + ([8] * 7)
                        + [EOS_ID]
                        + ([IGNORE_INDEX] * 3),
                    ]
                ),
            ),
            dict(
                input_ids=torch.tensor(
                    [
                        ([PAD_ID] * 5) + ([1] * 10) + ([PAD_ID] * 8),
                        ([PAD_ID] * 8) + ([2] * 6) + ([PAD_ID] * 9),
                        ([PAD_ID] * 9) + ([3] * 14),
                        ([PAD_ID] * 12) + [4, 4] + ([PAD_ID] * 9),
                        ([5] * 16) + ([PAD_ID] * 7),
                        ([PAD_ID] * 12) + ([6] * 8) + ([PAD_ID] * 3),
                        ([PAD_ID] * 9) + ([7] * 5) + ([PAD_ID] * 9),
                        ([PAD_ID] * 7) + ([8] * 13) + ([PAD_ID] * 3),
                    ]
                ),
                targets=torch.tensor(
                    [
                        [1, 1, EOS_ID] + ([IGNORE_INDEX] * 8),
                        [2, EOS_ID] + ([IGNORE_INDEX] * 9),
                        ([3] * 10) + [EOS_ID],
                        [4, EOS_ID] + ([IGNORE_INDEX] * 9),
                        [5, 5, 5, EOS_ID] + ([IGNORE_INDEX] * 7),
                        ([6] * 7) + [EOS_ID] + ([IGNORE_INDEX] * 3),
                        [7, EOS_ID] + ([IGNORE_INDEX] * 9),
                        ([8] * 7) + [EOS_ID] + ([IGNORE_INDEX] * 3),
                    ]
                ),
            ),
        ),
    ]


@pytest.mark.parametrize("batch, transformed_batch", args_sft_batch_transform())
def test_sft_batch_transform(batch, transformed_batch):
    batch_transform = SFTBatchTransform(
        eos_id=EOS_ID,
        ignore_index=IGNORE_INDEX,
        pad_id=PAD_ID,
    )
    batch_tr = batch_transform(batch)
    for k, v1 in transformed_batch.items():
        assert k in batch_tr
        v2 = batch_tr[k]
        print("\n" + k + "\n")
        print(str(v1) + "\n")
        print(v2)
        torch.testing.assert_close(v1, v2)


def test_sft_batch_transform_examples():
    questions = [
        "Who are you?",
        "How old are you?",
        "WTF?",
        "How many coffees did you drink today?",
        "Why is the sky blue?",
        "Could you repeat this?",
    ]
    answers = [
        "I am a purple unicorn",
        "My age is 27",
        "My pleasure",
        "1",
        "I must have slept in physics",
        "No, I will not do that",
    ]
    batch_transform = SFTBatchTransform(
        eos_id=EOS_ID,
        ignore_index=IGNORE_INDEX,
        pad_id=PAD_ID,
    )
    # Input to batch transform
    q_tokens = [[ord(c) for c in row] for row in questions]
    a_tokens = [[ord(c) for c in row] + [EOS_ID] for row in answers]
    max_q = max(len(q) for q in q_tokens)
    max_a = max(len(a) for a in a_tokens)
    max_qa = max(len(q) + len(a) for q, a in zip(q_tokens, a_tokens))
    input_ids = [
        q + a + [PAD_ID] * (max_qa - len(q) - len(a))
        for q, a in zip(q_tokens, a_tokens)
    ]
    labels = []
    for q, a in zip(q_tokens, a_tokens):
        len_q = len(q)
        len_qa = len(a) + len_q
        labels.append([IGNORE_INDEX] * len(q) + a + [IGNORE_INDEX] * (max_qa - len_qa))
    batch = {
        INPUT_IDS_NAME: torch.tensor(input_ids),
        LABELS_NAME: torch.tensor(labels),
    }
    # Desired output
    input_ids = []
    targets = []
    for q, a in zip(q_tokens, a_tokens):
        len_q = len(q)
        len_a = len(a)
        a_stripped = a[:-1]
        l_pad = max_q - len_q
        r_pad = max_a - len_a
        input_ids.append([PAD_ID] * l_pad + q + a_stripped + [PAD_ID] * r_pad)
        targets.append(a + [IGNORE_INDEX] * r_pad)
    # Compare
    batch_tr = batch_transform(batch)
    print(
        f"[input]:\n{INPUT_IDS_NAME}\n{batch[INPUT_IDS_NAME]}\n"
        f"{LABELS_NAME}\n{batch[LABELS_NAME]}\n"
    )
    for name, desired in (
        ("input_ids", input_ids),
        ("targets", targets),
    ):
        result = batch_tr[name]
        print(f"* {name}:\nresult:\n{result}\ndesired:\n{torch.tensor(desired)}")
        for i, (r_row, d_row) in enumerate(zip(result, desired)):
            r_row = r_row.tolist()
            assert r_row == d_row, (i, r_row, d_row)
