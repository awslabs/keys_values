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
import json
from typing import List, Literal, Counter, Callable

from keys_values.evaluation.response_parser import (
    extract_numbers,
    extract_choice_letter,
    extract_value_token,
)
from keys_values.evaluation.metrics import (
    sub_exact_match,
    exact_match,
    ndcg_at_10_ranked_numbers,
    rouge_n_f1,
)


def _eval_rag(responses: List[str], targets: List[List]) -> float:
    """
    Compute the SubEM score for the outputs for an RAG task

    Args:
        responses: a list of output strings
        targets: a list of target, while the target is a list of possible strings
    """
    scores = []
    for resp, tgt in zip(responses, targets):
        score = 0.0
        for t in tgt:
            if sub_exact_match(resp, t):
                score = 1.0
                break
        scores.append(score)
    return sum(scores) / len(scores)


def _eval_rerank(responses: List[str], targets: List[str]) -> float:
    """
    Comput the NDCG@10 score for the outputs for a rerank task

    Args:
        responses: a list of output strings
        targets: a list of target strings
    """
    scores = []
    for resp, tgt in zip(responses, targets):
        resp = extract_numbers(resp)
        tgt = [t.strip() for t in tgt.split(">")]
        score = ndcg_at_10_ranked_numbers(resp, tgt)
        scores.append(score)
    return sum(scores) / len(scores)


def _eval_icl(responses: List[str], targets: List[str]) -> float:
    """
    Compute the Exact Match score for the outputs for an in-context learning task

    Args:
        responses: a list of output strings
        targets: a list of target strings
    """
    scores = []
    for resp, tgt in zip(responses, targets):
        resp = Counter(extract_numbers(resp)).most_common(1)[0][0]
        score = 1.0 if exact_match(resp, tgt) else 0.0
        scores.append(score)
    return sum(scores) / len(scores)


def _eval_narrative_qa(accuracies: List[float]):
    """
    Compute the model-based score for narrative QA task

    Args:
        accuracies: a list of accuracy scores
    """
    return sum(accuracies) / len(accuracies)


def _eval_infinite_qa(responses: List[str], targets: List[str]):
    """
    Compute the Rouge-N F1 score for the outputs for an infinite qa task

    Args:
        responses: a list of output strings
        targets: a list of target strings
    """
    scores = []
    for resp, tgt in zip(responses, targets):
        score = rouge_n_f1(resp, tgt)
        scores.append(score)

    return sum(scores) / len(scores)


def _eval_infinite_mc(responses: List[str], targets: List[str]) -> float:
    """
    Compute the Exact Match score for the outputs for an infinite multiple choice task

    Args:
        responses: a list of output strings
        targets: a list of target strings
    """
    scores = []
    for resp, tgt in zip(responses, targets):
        resp = extract_choice_letter(resp)
        score = 1.0 if exact_match(resp, tgt) else 0.0
        scores.append(score)

    return sum(scores) / len(scores)


def _eval_summarization(
    fluency: List[int], precision: List[float], recall: List[float]
) -> float:
    """
    Compute the model-based score for summarization task
    Args:
        fluency: a list of fluency scores
        precision: a list of precision scores
        recall: a list of recall scores
    """
    scores = []
    for fl, pr, rc in zip(fluency, precision, recall):
        score = fl * (2 * pr * rc) / (pr + rc) if (pr + rc) > 0 else 0
        scores.append(score)

    return sum(scores) / len(scores)


def _eval_synthetic(
    responses: List[str],
    targets: List[str] | List[List],
    extract_func: Callable[[str], str],
) -> float:
    """
    Compute the SubEM score for the outputs for a synthetic task

    Args:
        responses: a list of output strings
        targets: a list of target strings or a list of target, while the target is a list of possible strings
        extract_func: the extraction function, which can be either extract_numbers or extract_value_token
    """
    scores = []
    for resp, tgt in zip(responses, targets):
        if isinstance(tgt, str):
            tgt = [tgt]
        resp = extract_func(resp)
        if isinstance(resp, str):
            resp = [resp]
        score = min(len(set(resp).intersection(tgt)) / len(tgt), 1)
        scores.append(score)

    return sum(scores) / len(scores)


def load_responses(
    eval_dataset_name: str,
    train_dataset_name: str,
    eval_len: Literal["8k", "16k", "32k", "64k", "128k"] = "16k",
    train_len: Literal["16k", "32k", "64k", "128k"] = "16k",
    results_dir="",
):
    """
    Extract the data from the results directory

    Args:
        eval_dataset_name: the name of the evaluation dataset
        train_dataset_name: the name of the training dataset, "base" means the model is not trained
        eval_len: the length of the evaluation dataset
        train_len: the length of the training dataset
        results_dir: the directory to store all results
    """
    # if train_dataset_name != "base":
    #     result_path = f"{results_dir}/{train_dataset_name}-{train_len}/{eval_len}/eval_{eval_len}.jsonl"
    # else:
    #     result_path = f"{results_dir}/base/{eval_len}/eval_{eval_len}.jsonl"
    if train_dataset_name != "base":
        result_path = (
            f"{results_dir}/{train_dataset_name}-{train_len}/rag_{eval_len}.jsonl"
        )
    else:
        result_path = f"{results_dir}/base/rag_{eval_len}.jsonl"

    instances = []
    with open(result_path, "r") as f:
        for line in f:
            instance = json.loads(line)
            if instance["req_id"].split("-")[0] == eval_dataset_name:
                instances.append(instance)
    return instances


def load_model_based_results(
    eval_dataset_name: str,
    train_dataset_name: str,
    eval_len: Literal["8k", "16k", "32k", "64k", "128k"] = "16k",
    train_len: Literal["16k", "32k", "64k", "128k"] = "16k",
    model_based_results_dir="",
    metric: Literal["accuracy", "fluency", "precision", "recall"] = "accuracy",
):
    """
    This function is used to load the model-based responses
    """
    data_dir = f"{model_based_results_dir}/{metric}"
    if train_dataset_name != "base":
        # this path is temporarily hardcoded, so please follow the script provided to generate prompts, send batches, and then download results
        data_path = f"{data_dir}/responses_eval_{eval_len}_parsed.jsonl"
        source_key = (
            f"{train_dataset_name}-{train_len}/eval_{eval_len}/{eval_dataset_name}"
        )
    else:
        data_path = f"{data_dir}/responses_eval_base_parsed.jsonl"
        source_key = f"base/eval_{eval_len}/{eval_dataset_name}"

    with open(data_path, "r") as f:
        instances = []
        for line in f:
            instance = json.loads(line)
            if source_key in instance["source"]:
                instances.append(instance)

        metric_scores = []
        if metric == "accuracy":
            for item in instances:
                fluency = (
                    item["fluency_correctness"][0]
                    if item["fluency_correctness"][0]
                    else 0
                )
                correctness = (
                    item["fluency_correctness"][1]
                    if item["fluency_correctness"][1]
                    else 0
                )
                metric_scores.append(fluency * correctness / 3)
        elif metric == "fluency":
            for item in instances:
                fluency = item["fluency"] if item["fluency"] else 0
                metric_scores.append(fluency)
        elif metric == "precision":
            for item in instances:
                precision_count = (
                    item["precision_sentence_count"][0]
                    if item["precision_sentence_count"][0]
                    else 0
                )
                sentence_count = (
                    item["precision_sentence_count"][1]
                    if item["precision_sentence_count"][1]
                    else 0
                )
                metric_scores.append(
                    precision_count / sentence_count if sentence_count != 0 else 0
                )
        elif metric == "recall":
            for item in instances:
                recall_count = (
                    item["recall_keypoints_count"][0]
                    if item["recall_keypoints_count"][0]
                    else 0
                )
                keypoints_count = (
                    item["recall_keypoints_count"][1]
                    if item["recall_keypoints_count"][1]
                    else 0
                )
                metric_scores.append(
                    recall_count / keypoints_count if keypoints_count != 0 else 0
                )
        else:
            raise ValueError("Unknown metric")
    return metric_scores


def evaluation_score(
    eval_dataset_name: str,
    train_dataset_name: str,
    eval_len: Literal["8k", "16k", "32k", "64k", "128k"] = "16k",
    train_len: Literal["16k", "32k", "64k", "128k"] = "16k",
    results_dir="",
    model_based_results_dir="",
):
    """
    Compute the evaluation score for a given evaluation dataset and model training configuration

    Args:
        eval_dataset_name: the name of the evaluation dataset
        train_dataset_name: the name of the training dataset, "base" means the model is not trained
        eval_len: the length of the evaluation dataset
        train_len: the length of the training dataset
        results_dir: the directory to store all results
        model_based_results_dir: the directory to store all model-based results
    """
    result_data = load_responses(
        eval_dataset_name, train_dataset_name, eval_len, train_len, results_dir
    )
    responses = []
    response_lengths = []
    targets = []
    for item in result_data:
        responses.append(item["output_text"])
        targets.append(item["target"])
        response_lengths.append(item["output_tokens"])

    if eval_dataset_name in ["nq", "trivia_qa", "pop_qa", "hotpot_qa"]:
        eval_score = _eval_rag(responses, targets)
    elif eval_dataset_name in ["ms_macro"]:
        eval_score = _eval_rerank(responses, targets)
    elif eval_dataset_name in [
        "trec_coarse",
        "trec_fine",
        "nlu",
        "banking77",
        "clinc150",
    ]:
        eval_score = _eval_icl(responses, targets)
    elif eval_dataset_name in ["narrative_qa"]:
        metric_scores = load_model_based_results(
            eval_dataset_name,
            train_dataset_name,
            eval_len,
            train_len,
            model_based_results_dir,
            metric="accuracy",
        )
        eval_score = _eval_narrative_qa(metric_scores)
    elif eval_dataset_name in ["infinite_bench_qa"]:
        eval_score = _eval_infinite_qa(responses, targets)
    elif eval_dataset_name in ["infinite_bench_mc"]:
        eval_score = _eval_infinite_mc(responses, targets)
    elif eval_dataset_name in ["infinite_bench_sum", "multi_lex_sum"]:
        fluency_scores = load_model_based_results(
            eval_dataset_name,
            train_dataset_name,
            eval_len,
            train_len,
            model_based_results_dir,
            metric="fluency",
        )
        precision_scores = load_model_based_results(
            eval_dataset_name,
            train_dataset_name,
            eval_len,
            train_len,
            model_based_results_dir,
            metric="precision",
        )
        recall_scores = load_model_based_results(
            eval_dataset_name,
            train_dataset_name,
            eval_len,
            train_len,
            model_based_results_dir,
            metric="recall",
        )
        eval_score = _eval_summarization(
            fluency_scores, precision_scores, recall_scores
        )
    elif eval_dataset_name in ["json_kv", "ruler_mk_uuid"]:
        eval_score = _eval_synthetic(
            responses, targets, extract_func=extract_value_token
        )
    elif eval_dataset_name in ["ruler_mk_needle", "ruler_mv"]:
        eval_score = _eval_synthetic(responses, targets, extract_func=extract_numbers)
    else:
        raise NotImplementedError(
            f"Evaluation for {eval_dataset_name} is not implemented yet."
        )

    average_len = sum(response_lengths) / len(response_lengths)

    return eval_score, average_len
