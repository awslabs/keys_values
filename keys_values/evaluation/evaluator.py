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
from keys_values.kvcache.replay_buffers import ModelForTokenGeneration
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


def _compute_metric(
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

    Up to `max_generated_tokens` tokens are generated for each batch
    position. In each batch position, generation is stopped once
    `tokenizer.eos_id` is drawn.
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
        prompts_or_logits: torch.Tensor,
        targets: List[TargetType],
    ) -> Dict[str, torch.Tensor]:
        """
        Computes metric values for data case.

        Args:
            model: LongContextInferenceModel
            prompts_or_logits: Either prompts, shape `(batch_size,
                prompt_length)`, or logits of final token_position, shape
                `(batch_size, 1, padded_vocab_size)`. In the latter case, we
                skip prompt processing, assuming the KV caches have been
                prepared appropriately.
            targets: List of targets of length `batch_size`. Each entry is a
                string or list of strings. Some metrics allow for lists of
                strings, others require a single string

        Returns:
            Dictionary with entries `{name: values}`, where `name in self.metrics`
            and `values.shape = (batch_size,)`, the metric values for each
            entry in the batch.

        """
        assert 2 <= prompts_or_logits.ndim == 3
        batch_size = prompts_or_logits.shape[0]
        if len(targets) != batch_size:
            raise ValueError(
                f"len(targets) = {len(targets)} != {batch_size} = batch_size"
            )
        for target in targets:
            for metric in self.metrics:
                validate_targets(target, metric)

        # Generate tokens
        generated_tokens = torch.cat(
            list(
                batched_generate_fn(
                    model=model,
                    prompts_or_logits=prompts_or_logits,
                    max_returned_tokens=self.max_generated_tokens,
                    ignore_index=self._eos_id,
                    sample_args=self.sample_kwargs,
                    stop_tokens=([self._eos_id],),
                )
            ),
            dim=-1,
        )
        outputs = [
            self.tokenizer.decode(seq[seq != self._eos_id]) for seq in generated_tokens
        ]
        assert len(outputs) == batch_size, (outputs, batch_size)
        return {
            metric: torch.tensor(
                [
                    _compute_metric(output, target, metric)
                    for output, target in zip(outputs, targets)
                ],
                dtype=torch.float32,
                device=prompts_or_logits.device,
            )
            for metric in self.metrics
        }


def compute_sample_based_metrics(
    model: LongContextInferenceModel,
    evaluator: SampleBasedMetricsEvaluator,
    gen_wrapper: ModelForTokenGeneration,
    input_ids: torch.Tensor,
    targets: Optional[torch.Tensor],
    raw_targets: List[TargetType],
    num_samples: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Computes sample-based metrics specified by `evaluator`, using replay
    buffers during generation. The buffer switching is managed by
    `gen_wrapper`.

    Args:
        model: LongContextInferenceModel
        evaluator: Computes sample-based metrics
        gen_wrapper: Manages buffer switching
        input_ids: Inputs. If `targets == None`, these are the prompts
        targets: If given, forms a data case with `input_ids`, in which
            case `model.forward` is called with `mode="inputs"`, so that
            the loss value can be computed afterwards. If not given,
            `input_ids` are the prompts.
        raw_targets: Raw targets (not tokenized) passed into `evaluator`
        num_samples: Sample-based metrics are averaged over this many
            token generations.

    Returns:
        Dictionary with entries `{name: values}`, where `name in
        evaLuator.metrics` and `values.shape = (batch_size,)`, the metric
        values for each entry in the batch. If `num_samples > 1`, the
        values are averaged over this many samples.

    """
    if num_samples < 1:
        raise ValueError(f"num_samples = {num_samples}, must be >= 1")
    if not(model.gpt_model is gen_wrapper.gpt_model):
        raise ValueError("model.gpt_model and gen_wrapper.gpt_model must be the same")
    with torch.no_grad():
        logits = model(
            input_ids,
            targets,
            mode="both" if targets is None else "inputs",
        )
    result = None
    for _ in range(num_samples):
        gen_wrapper.switch_status(True)
        metric_vals = evaluator(
            model=model,
            prompts_or_logits=logits,
            targets=raw_targets,
        )
        gen_wrapper.switch_status(False)
        if result is None:
            result = metric_vals
        else:
            for k, v in metric_vals.items():
                result[k] += v
    if num_samples > 1:
        for k, v in result.items():
            v /= num_samples
    return result
