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
from typing import Any, Dict, Optional, List, Union

import torch

from litgpt.tokenizer import Tokenizer

from keys_values.evaluation.metrics import sub_exact_match, rouge_n_f1
from keys_values.evaluation.evaluation import _eval_rerank, _eval_icl, _eval_infinite_mc
from keys_values.generate.base import batched_generate_fn
from keys_values.long_context import LongContextInferenceModel

METRICS_FOR_HELMET_TASKS = {
    "nq": "sub_exact_match",
    "trivia_qa": "sub_exact_match",
    "pop_qa": "sub_exact_match",
    "hotpot_qa": "sub_exact_match",
    "ms_macro": "ndcg_at_10",
    "trec_coarse": "exact_match",
    "trec_fine": "exact_match",
    "nlu": "exact_match",
    "banking77": "exact_match",
    "clinc150": "exact_match",
    "infinite_bench_qa": "rouge_n_f1",
    "infinite_bench_mc": "infinite_mc",
}

TargetType = Union[List[str], str]


def validate_targets(targets: TargetType, metric: str):
    is_list_str = isinstance(targets, list) and all(isinstance(x, str) for x in targets)
    is_str = isinstance(targets, str) or (is_list_str and len(targets) == 1)
    if metric == "sub_exact_match" and not (is_list_str or is_str):
        raise ValueError(
            f"Metric {metric} needs list of string targets, got: {targets}"
        )
    if metric != "sub_exact_match" and not is_str:
        raise ValueError(f"Metric {metric} needs string targets, got: {targets}")


def compute_metric(
    output: str,
    targets: TargetType,
    metric: str,
) -> float:
    if metric == "sub_exact_match":
        if isinstance(targets, list):
            return float(any(sub_exact_match(output, target) for target in targets))
        else:
            return float(sub_exact_match(output, targets))
    elif metric == "ndcg_at_10":
        return _eval_rerank([output], [targets])
    elif metric == "exact_match":
        return _eval_icl([output], [targets])
    elif metric == "rouge_n_f1":
        return rouge_n_f1(output, targets)
    elif metric == "exact_match":
        return _eval_infinite_mc([output], [targets])
    else:
        raise ValueError(f"Metric {metric} not supported")


class SampleBasedMetricsEvaluator:
    """
    Evaluates metrics which depend on generating a maximum number of
    tokens.

    """

    def __init__(
        self,
        metrics: List[str],
        max_generated_tokens: int,
        tokenizer: Tokenizer,
        sample_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not all(metric in self.supported_metrics() for metric in metrics):
            raise ValueError(f"Metrics {metrics} not all supported")
        self.metrics = metrics
        self.max_generated_tokens = max_generated_tokens
        if sample_kwargs is None:
            sample_kwargs = dict()
        else:
            sample_kwargs = sample_kwargs.copy()
        self.tokenizer = tokenizer
        self._eos_id = tokenizer.eos_id
        self.sample_kwargs = sample_kwargs

    @staticmethod
    def supported_metrics() -> List[str]:
        """
        Returns:
            List of names of supported metrics

        """
        return list(METRICS_FOR_HELMET_TASKS.values())

    @staticmethod
    def metric_for_helmet_task(dataset_key: str) -> Optional[str]:
        """
        Args:
            dataset_key: Name of Helmet dataset

        Returns:
            Evaluation metric to use for this task; or `None` if metric is
            not supported here for this task, or `dataset_key` is invalid.

        """
        return METRICS_FOR_HELMET_TASKS.get(dataset_key)

    def __call__(
        self,
        model: LongContextInferenceModel,
        prompts: torch.Tensor,
        targets: List[TargetType],
    ) -> Dict[str, torch.Tensor]:
        """
        Computes metric values for data case `(input_ids, targets)`. The
        metrics to be computed are in `metrics`.

        Args:
            model: LongContextInferenceModel
            prompts: Prompts, `(batch_size, prompt_len)`. Aligned on the
                right (if there is padding, it is on the left)
            targets: List of targets of length `batch_size`. Each entry is a
                string or list of strings. Some metrics allow for lists of
                strings, others require a single string

        Returns:
            Dictionary with entries `{name: values}`, where `name in self.metrics`
            and `values.shape = (batch_size,)`, the metric values for each
            entry in the batch.

        """

        def right_pad(elem: List[Union[torch.Tensor, None]]) -> List[torch.Tensor]:
            lens = [0 if s is None else s.shape[0] for s in elem]
            max_len = max(lens)
            assert max_len > 0, (elem,)
            x = next(s for s in elem if s is not None)
            kwargs = dict(fill_value=self._eos_id, dtype=x.dtype, device=x.device)
            pads = [torch.full((max_len - l,), **kwargs) for l in lens]
            return [p if s is None else torch.cat((s, p)) for s, p in zip(elem, pads)]

        assert prompts.ndim == 2
        batch_size = prompts.shape[0]
        if len(targets) != batch_size:
            raise ValueError(
                f"len(targets) = {len(targets)} != {batch_size} = batch_size"
            )
        for target in targets:
            for metric in self.metrics:
                validate_targets(target, metric)

        # Generate tokens
        parts = [
            right_pad(elem)
            for elem in batched_generate_fn(
                model=model,
                prompts=prompts,
                max_generated_tokens=self.max_generated_tokens,
                sample_kwargs=self.sample_kwargs,
                stop_tokens=([self._eos_id],),
                include_prompt=False,
            )
        ]
        outputs = [self.tokenizer.decode(torch.cat(elems)) for elems in zip(*parts)]
        assert len(outputs) == batch_size, (outputs, batch_size)
        return {
            metric: torch.tensor(
                [
                    compute_metric(output, target, metric)
                    for output, target in zip(outputs, targets)
                ],
                dtype=torch.float32,
                device=prompts.device,
            )
            for metric in self.metrics
        }
