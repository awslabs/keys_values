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

from keys_values.finetune.batch_transform import SFTBatchTransform


PAD_ID = 0

IGNORE_INDEX = -100


def args_sft_batch_transform():
    return [
        (
            dict(
                input_ids=torch.tensor(
                    [
                        [ 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        [10, 11, 12, 13, 14, 15, 16, 17, PAD_ID],
                    ]
                ),
                labels = torch.tensor(
                    [
                        [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 8, 9],
                        [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 14, 15, 16, 17, IGNORE_INDEX],
                    ]
                )
            ),
            dict(
                input_ids=torch.tensor(
                    [
                        [ 1, 2, 3, 4, 5, 6, 7, 8, 9, PAD_ID],
                        [PAD_ID, PAD_ID, PAD_ID, 10, 11, 12, 13, 14, 15, 16],
                    ]
                ),
                targets = torch.tensor(
                    [
                        [8, 9, IGNORE_INDEX, IGNORE_INDEX],
                        [14, 15, 16, 17],
                    ]
                ),
            )
        ),
        (
            dict(
                input_ids=torch.tensor(
                    [
                        ([1] * 10) + ([PAD_ID] * 6),
                        ([2] * 6) + ([PAD_ID] * 10),
                        ([3] * 14) + ([PAD_ID] * 2),
                        [4]  + ([PAD_ID] * 15),
                        [5] * 16,
                        ([6] * 8) + ([PAD_ID] * 8),
                        ([7] * 5) + ([PAD_ID] * 11),
                        ([8] * 13) + ([PAD_ID] * 3),
                    ]
                ),
                labels=torch.tensor(
                    [
                        ([IGNORE_INDEX] * 8) + ([9] * 2) + ([IGNORE_INDEX] * 6),
                        ([IGNORE_INDEX] * 5) + [10] + ([IGNORE_INDEX] * 10),
                        ([IGNORE_INDEX] * 4) + ([11] * 10) + ([IGNORE_INDEX] * 2),
                        [12] + ([IGNORE_INDEX] * 15),
                        ([IGNORE_INDEX] * 13) + [13] * 3,
                        ([14] * 8) + ([IGNORE_INDEX] * 8),
                        ([IGNORE_INDEX] * 4) + [15] + ([IGNORE_INDEX] * 11),
                        ([IGNORE_INDEX] * 6) + ([16] * 7) + ([IGNORE_INDEX] * 3),
                    ]
                )
            ),
            dict(
                input_ids=torch.tensor(
                    [
                        ([PAD_ID] * 5) + ([1] * 10) + ([PAD_ID] * 7),
                        ([PAD_ID] * 8) + ([2] * 6) + ([PAD_ID] * 8),
                        ([PAD_ID] * 9) + ([3] * 13),
                        ([PAD_ID] * 13) + [4] + ([PAD_ID] * 8),
                        [5] * 16 + ([PAD_ID] * 6),
                        ([PAD_ID] * 13) + ([6] * 8) + ([PAD_ID] * 1),
                        ([PAD_ID] * 9) + ([7] * 5) + ([PAD_ID] * 8),
                        ([PAD_ID] * 7) + ([8] * 13) + ([PAD_ID] * 2),
                    ]
                ),
                targets=torch.tensor(
                    [
                        ([9] * 2) + ([IGNORE_INDEX] * 8),
                        [10] + ([IGNORE_INDEX] * 9),
                        ([11] * 10),
                        [12] + ([IGNORE_INDEX] * 9),
                        [13] * 3 + ([IGNORE_INDEX] * 7),
                        ([14] * 8) + ([IGNORE_INDEX] * 2),
                        [15] + ([IGNORE_INDEX] * 9),
                        ([16] * 7) + ([IGNORE_INDEX] * 3),
                    ]
                ),
            )
        ),
    ]


@pytest.mark.parametrize("batch, transformed_batch", args_sft_batch_transform())
def test_sft_batch_transform(batch, transformed_batch):
    batch_transform = SFTBatchTransform(ignore_index=IGNORE_INDEX, pad_id=PAD_ID)
    batch_tr = batch_transform(batch)
    for k, v1 in transformed_batch.items():
        assert k in batch_tr
        v2 = batch_tr[k]
        print("\n" + k + "\n")
        print(str(v1) + "\n")
        print(v2)
        torch.testing.assert_close(v1, v2)
