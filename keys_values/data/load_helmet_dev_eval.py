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

import hashlib
import json
import math
import os
import shutil
import tarfile
import tempfile
import zipfile
from functools import partial
from typing import Literal
import random
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import requests

import numpy as np
from datasets import DatasetDict, Dataset, Value, Sequence, Features, enable_progress_bars, load_from_disk, load_dataset
from transformers import AutoTokenizer

enable_progress_bars()  # make sure bars aren't globally disabled
load_dotenv()  # reads .env into environment
HF_TOKEN = os.getenv("LLAMA3_MODEL_TOKEN")  # place you huggingface token here
DATASET_PARENT_DIR = "~/.cache/huggingface/helmet/data" # the default place to store the cache data

def download_source_data(download_dir: str) -> None:
    """
    Downloads + extracts HELMET data.tar.gz into `dataset_parent_dir`, deletes the archive,
    then copies all files from HELMET GitHub repo `prompts/` into the extracted `alce/` folder.
    Shows step messages + progress bars (tqdm if installed).
    """

    def step(i: int, n: int, msg: str) -> None:
        print(f"\n[Step {i}/{n}] {msg}")

    def download_with_progress(url: str, dst_path: Path, desc: str, timeout: int = 300) -> None:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dst_path, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=desc) as bar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

    def _is_within_directory(directory: Path, target: Path) -> bool:
        directory = directory.resolve()
        target = target.resolve()
        return str(target).startswith(str(directory) + os.sep) or target == directory

    def safe_extract_tar_with_progress(tar_path: Path, out_dir: Path) -> None:
        with tarfile.open(tar_path, "r:gz") as tf:
            members = tf.getmembers()

            # safety check (path traversal)
            for m in members:
                member_path = out_dir / m.name
                if not _is_within_directory(out_dir, member_path):
                    raise RuntimeError(f"Unsafe path in tar (path traversal): {m.name}")

            # extract with progress
            with tqdm(total=len(members), desc="Extracting data.tar.gz", unit="file") as bar:
                for m in members:
                    tf.extract(m, out_dir)
                    bar.update(1)

    def extract_zip_with_progress(zip_path: Path, out_dir: Path) -> None:
        with zipfile.ZipFile(zip_path) as zf:
            infos = zf.infolist()
            with tqdm(total=len(infos), desc="Extracting repo zip", unit="file") as bar:
                for info in infos:
                    zf.extract(info, out_dir)
                    bar.update(1)

    # ---------- main workflow ----------
    TOTAL_STEPS = 6
    download_dir = Path(download_dir)

    step(1, TOTAL_STEPS, f"Ensure dataset directory exists: {download_dir}")
    download_dir.mkdir(parents=True, exist_ok=True)

    data_url = "https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/main/data.tar.gz"
    repo_zip_url = "https://github.com/princeton-nlp/HELMET/archive/refs/heads/main.zip"

    step(2, TOTAL_STEPS, "Download dataset archive (data.tar.gz)")
    tar_gz_path = download_dir / "data.tar.gz"
    download_with_progress(data_url, tar_gz_path, desc="Downloading data.tar.gz")

    step(3, TOTAL_STEPS, "Extract dataset archive safely + delete archive")
    try:
        safe_extract_tar_with_progress(tar_gz_path, download_dir)
    finally:
        if tar_gz_path.exists():
            tar_gz_path.unlink()

    step(4, TOTAL_STEPS, "Locate extracted root and create alce/ directory")
    extracted_root = download_dir / "data"
    if not extracted_root.is_dir():
        extracted_root = download_dir
    alce_dir = extracted_root / "alce"
    alce_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Using extracted_root = {extracted_root}")
    print(f"  alce_dir = {alce_dir}")

    step(5, TOTAL_STEPS, "Download HELMET repo zip (for prompts/)")
    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        repo_zip_path = td / "HELMET-main.zip"
        download_with_progress(repo_zip_url, repo_zip_path, desc="Downloading HELMET repo zip")

        step(6, TOTAL_STEPS, "Extract repo zip + copy prompts/* into alce/prompts")
        extract_zip_with_progress(repo_zip_path, td)

        prompts_dir = td / "HELMET-main" / "prompts"
        if not prompts_dir.is_dir():
            raise FileNotFoundError(f"Couldn't find prompts dir at: {prompts_dir}")

        prompt_files = [p for p in prompts_dir.iterdir() if p.is_file()]
        os.makedirs(alce_dir / "prompts", exist_ok=True)
        for p in tqdm(prompt_files, desc="Copying prompts", unit="file"):
            shutil.copy2(p, alce_dir / "prompts" / p.name)
    print("\nThe source data preparation is done.")

def load_helmet_dev_eval(
        dataset_key,
        max_length: Literal["8k", "16k", "32k", "64k", "128k"] = "8k",
        dataset_parent_dir: str = DATASET_PARENT_DIR
):
    """
    Loads HELMET dev/eval data from disk or generates it from scratch.

    Args:
        dataset_key: The name of the dataset to load.
        max_length: The maximum length of the dataset.
        dataset_parent_dir: The parent directory where the dataset is stored.

    Returns:
        A tuple of (dev_data, eval_data) datasets. Each data instance will contain at least "input", "output", "query_id", "max_length" fields.
    """
    dataset_parent_dir = Path(dataset_parent_dir)
    source_data_dir = dataset_parent_dir.parent
    # 1) If the source data does not exisit, download it first
    if not os.path.exists(source_data_dir):
        download_source_data(source_data_dir)

    cache_dir = os.path.join(dataset_parent_dir.parent, f"longtrain_{dataset_key}_{max_length}")

    # 2) If cached, load and return
    if os.path.isdir(cache_dir):
        dsd = load_from_disk(cache_dir)  # expects a DatasetDict with dev/val
        # Support either naming convention if you ever change it
        if "development" in dsd and "evaluation" in dsd:
            return dsd["development"], dsd["evaluation"]
        raise ValueError(f"Cache found at {cache_dir}, but it doesn't contain expected splits.")

    # 3) Otherwise generate using your existing logic
    if dataset_key in ["nq", "trivia_qa", "hotpot_qa", "pop_qa"]:
        dev_data, eval_data = load_rag(dataset_key, max_length, dataset_parent_dir)
    elif dataset_key in ["alce_asqa", "alce_qampari"]:
        dev_data, eval_data = load_cited_generation(dataset_key, max_length, dataset_parent_dir)
    elif dataset_key in ["ms_macro"]:
        dev_data, eval_data = load_rerank(dataset_key, max_length, dataset_parent_dir)
    elif dataset_key in ["trec_coarse", "trec_fine", "nlu", "banking77", "clinc150"]:
        dev_data, eval_data = load_icl(dataset_key, max_length)
    elif dataset_key in ["narrative_qa", "infinite_bench_qa", "infinite_bench_mc"]:
        dev_data, eval_data = load_long_doc_qa(dataset_key, max_length)
    elif dataset_key in ["infinite_bench_sum", "multi_lex_sum"]:
        dev_data, eval_data = load_summarization(dataset_key, max_length)
    elif dataset_key in ["json_kv", "ruler_mk_needle", "ruler_mk_uuid", "ruler_mv"]:
        dev_data, eval_data = load_synthetic(dataset_key, max_length, dataset_parent_dir)
    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    # 4) Save for next time
    os.makedirs(dataset_parent_dir, exist_ok=True)
    DatasetDict({"development": dev_data, "evaluation": eval_data}).save_to_disk(cache_dir)

    return dev_data, eval_data

def drop_duplicates(
    data,
    instance_key: str = "id",
    shuffle: bool = False,
    seed: int | None = 42,
):
    # Optional shuffle to make which row is kept random
    if shuffle:
        data = data.shuffle(seed=seed)

    indices_to_keep = []
    seen = set()

    # Iterate only over the relevant column for a tiny bit less overhead
    column = data[instance_key]
    for i, value in enumerate(column):
        if value in seen:
            continue
        seen.add(value)
        indices_to_keep.append(i)

    return data.select(indices_to_keep)

def load_rag(
        dataset_key: Literal["nq", "trivia_qa", "hotpot_qa", "pop_qa"] = "nq",
        max_length: Literal["8k", "16k", "32k", "64k", "128k"] = "8k",
        dataset_parent_dir: str = DATASET_PARENT_DIR
):
    """
    Load dataset the belongs to the RAG task
    """
    shots = 2
    eval_questions_num = 100    # the 100 questions that will be used in the evaluation set, aka, in HELMET
    popularity_threshold = 3    # this is only used by popqa, to filter the long-tail entities
    file_paths = {
        "nq": {"demo_file": "kilt/nq-train-multikilt_1000_k3_dep6.jsonl",
               "128k": "kilt/nq-dev-multikilt_1000_k1000_dep6.jsonl",
               "64k": "kilt/nq-dev-multikilt_1000_k440_dep6.jsonl",
               "32k": "kilt/nq-dev-multikilt_1000_k220_dep6.jsonl",
               "16k": "kilt/nq-dev-multikilt_1000_k105_dep6.jsonl",
               "8k": "kilt/nq-dev-multikilt_1000_k50_dep6.jsonl"},
        "trivia_qa": {"demo_file": "kilt/triviaqa-train-multikilt_1000_k3_dep6.jsonl",
                      "128k": "kilt/triviaqa-dev-multikilt_1000_k1000_dep6.jsonl",
                      "64k": "kilt/triviaqa-dev-multikilt_1000_k440_dep6.jsonl",
                      "32k": "kilt/triviaqa-dev-multikilt_1000_k220_dep6.jsonl",
                      "16k": "kilt/triviaqa-dev-multikilt_1000_k105_dep6.jsonl",
                      "8k": "kilt/triviaqa-dev-multikilt_1000_k50_dep6.jsonl"},
        "hotpot_qa": {"demo_file": "kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl",
                      "128k": "kilt/hotpotqa-dev-multikilt_1000_k1000_dep3.jsonl",
                      "64k": "kilt/hotpotqa-dev-multikilt_1000_k440_dep3.jsonl",
                      "32k": "kilt/hotpotqa-dev-multikilt_1000_k220_dep3.jsonl",
                      "16k": "kilt/hotpotqa-dev-multikilt_1000_k105_dep3.jsonl",
                      "8k": "kilt/hotpotqa-dev-multikilt_1000_k50_dep3.jsonl"},
        "pop_qa": {"demo_file": "kilt/popqa_test_1000_k3_dep6.jsonl",
                   "128k": "kilt/popqa_test_1000_k1000_dep6.jsonl",
                   "64k": "kilt/popqa_test_1000_k440_dep6.jsonl",
                   "32k": "kilt/popqa_test_1000_k220_dep6.jsonl",
                   "16k": "kilt/popqa_test_1000_k105_dep6.jsonl",
                   "8k": "kilt/popqa_test_1000_k50_dep6.jsonl"},
    }   # the load paths can only be stored in this way, as they are hard-coded from the original code
    instruction_template = "Use the given documents to write a concise and short answer to the question. Write your answer in the following format:\nAnswer: [answer]\n\n{demos}{context}\n\nQuestion: {question}\nAnswer:"

    instance_path = str(Path(dataset_parent_dir) / Path(file_paths[dataset_key][max_length]))
    demo_path = str(Path(dataset_parent_dir) / Path(file_paths[dataset_key]["demo_file"]))
    instance_data = load_dataset("json", data_files=instance_path)["train"]
    demon_data = load_dataset("json", data_files=demo_path)["train"]

    if dataset_key in ["nq", "trivia_qa", "hotpot_qa"]:
        eval_questions =  random.sample(sorted(set(instance_data["question"])), eval_questions_num)
        eval_data = instance_data.filter(lambda x: x["question"] in eval_questions)
        dev_data = instance_data.filter(lambda x: x["question"] not in eval_questions)
        drop_columns = ["question", "answers", "hard_negative_ctxs", "ctxs"]
    else:
        popularity_filtered = instance_data.filter(lambda x: math.log10(x['s_pop']) < popularity_threshold)
        eval_questions = random.sample(sorted(set(popularity_filtered["id"])), eval_questions_num)
        eval_data = popularity_filtered.filter(lambda x: x["id"] in eval_questions)
        dev_data = instance_data.filter(lambda x: x["id"] not in eval_questions)
        drop_columns = ["id", "subj", "prop", "obj", "subj_id", "prop_id", "obj_id", "s_aliases", "o_aliases", "s_uri",
                        "o_uri", "o_wiki_title", "s_wiki_title", "s_pop", "o_pop", "question", "answers", "ctxs"]

    def _rag_instruction_fillup(instance):
        question_id = "question" if dataset_key != "pop_qa" else "id"
        demos = demon_data if dataset_key != "pop_qa" else demon_data.filter(lambda x: x[question_id] != instance[question_id])
        demo_template = "{documents}\n\nQuestion: {question}\nAnswer: {answer}"
        passage_template = "Document (Title: {title}): {text}"

        h = int(hashlib.sha256(str(instance[question_id]).encode("utf-8")).hexdigest(), 16) % 2**31
        demos = demos.shuffle(seed=h)
        demos = drop_duplicates(demos, question_id).select(range(shots))
        demo_text = "\n\n".join([demo_template.format(**d, documents="\n\n".join(
            [passage_template.format(**c) for c in d["ctxs"]]), answer=d["answers"][0]) + "\n\n" for d in demos])
        context_text = ""
        if len(instance['ctxs']) > 0:
            context_text = "\n\n".join([passage_template.format(**c) for c in instance['ctxs']])

        instruction = instruction_template.format(demos=demo_text, context=context_text, question=instance["question"])
        return {"input": instruction, "output": instance["answers"],
                "query_id": instance["question"], "max_length": max_length}

    dev_data = drop_duplicates(dev_data, instance_key="question", shuffle=True)
    dev_data = dev_data.map(_rag_instruction_fillup, remove_columns=drop_columns)
    eval_data = eval_data.map(_rag_instruction_fillup, remove_columns=drop_columns)
    return dev_data, eval_data

def load_cited_generation(
        dataset_key: Literal["alce_asqa", "alce_qampari"] = "alce_asqa",
        max_length: Literal["8k", "16k", "32k", "64k", "128k"] = "8k",
        dataset_parent_dir: str = DATASET_PARENT_DIR
):
    """
    Load dataset the belongs to the generation with citation task
    Please be aware that the development partition cannot be used for training.
    """
    shots = 2
    seed=42
    eval_questions_num = 100  # the 100 questions that will be used in the evaluation set
    file_paths = {
        "alce_asqa": {"demo_file": "alce/prompts/asqa_revised.json",
               "test_file": "alce/asqa_eval_gtr_top2000.json"},
        "alce_qampari": {"demo_file": "alce/prompts/qampari_revised.json",
               "test_file": "alce/qampari_eval_gtr_top2000.json"},
    }
    num_docs_map = {"128k": 700, "64k": 345, "32k": 165, "16k": 75, "8k": 30,}
    instruction_template = "{demos}Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing a document, surround its ID with square brackets, such as [x] to cite document x. To cite multiple documents, simply concatenate the citation markers; for example, use [x][y][z] to cite the documents with ID x, y, and z. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.\n\nQuestion: {question}\n\n{context}\n\nAnswer:"
    demo_template = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing a document, surround its ID with square brackets, such as [x] to cite document x. To cite multiple documents, simply concatenate the citation markers; for example, use [x][y][z] to cite the documents with ID x, y, and z. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.\n\nQuestion: {question}\n\n{context}\n\nAnswer: {answer}"
    doc_template = "Document [{ID}](Title: {title}): {text}"

    instance_path = str(Path(dataset_parent_dir) / Path(file_paths[dataset_key]["test_file"]))
    demo_path = str(Path(dataset_parent_dir) / Path(file_paths[dataset_key]["demo_file"]))
    instance_data = load_dataset("json", data_files=instance_path)["train"]
    demo_data = json.load(open(demo_path))
    num_docs = num_docs_map[max_length]

    def _cg_instruction_fillup(instance):
        context = "\n\n".join(
            [doc_template.format(**d, ID=idx + 1) for idx, d in enumerate(instance["docs"][:num_docs])])
        demos = "\n\n\n".join([
            demo_template.format(**demo, context="\n\n".join(
                [doc_template.format(**d, ID=idx + 1) for idx, d in enumerate(demo["docs"])]))
            for demo in random.sample(demo_data["demos"], shots)
        ])
        return {"input": instruction_template.format(demos=demos, context=context, question=instance["question"]),
                "output": "", "query_id": instance["question"], "max_length": max_length}

    instance_data = instance_data.map(_cg_instruction_fillup).shuffle(seed=seed)
    eval_data = instance_data.select(range(eval_questions_num))
    dev_data = instance_data.select(range(eval_questions_num, len(instance_data)))
    return dev_data, eval_data

def load_rerank(
        dataset_key: Literal["ms_macro"] = "ms_macro",
        max_length: Literal["8k", "16k", "32k", "64k", "128k"] = "8k",
        dataset_parent_dir: str = DATASET_PARENT_DIR
):
    """Load dataset that belongs to the passage re-ranking task."""
    seed = 42
    shots = 2
    eval_questions_num = 40  # the 100 questions that will be used in the evaluation set
    file_paths = {
        "ms_macro": {"demo_file": "msmarco/test_reranking_data_k10_dep3.jsonl",
               "128k": "msmarco/test_reranking_data_k1000_dep3.jsonl",
               "64k": "msmarco/test_reranking_data_k600_dep3.jsonl",
               "32k": "msmarco/test_reranking_data_k285_dep3.jsonl",
               "16k": "msmarco/test_reranking_data_k130_dep3.jsonl",
               "8k": "msmarco/test_reranking_data_k50_dep3.jsonl"},
    }
    instruction_template = "You are provided with a list of documents, each indicated by their ID. Rank each document based on their relevance to the question in descending order from most relelvant to least relevant texts. Include all documents in the rankings. Write your answer using the unique IDs, with the following format:\nRanking: ID3 > ID1 > ID2\n\n{demos}{context}\n\nQuery: {question}\nRanking:"

    instance_path = str(Path(dataset_parent_dir) / Path(file_paths[dataset_key][max_length]))
    demo_path = str(Path(dataset_parent_dir) / Path(file_paths[dataset_key]["demo_file"]))
    instance_data = load_dataset("json", data_files=instance_path)["train"]
    demo_data = load_dataset("json", data_files=demo_path)["train"]

    k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]  # the k values that will be used to calculate metrics later
    k_values = [k for k in k_values if k <= len(instance_data[0]["ctxs"])]

    def _rerank_instance_fillup(instance):
        passage_template = "[ID: {id}] Document (Title: {title}): {text}" if "title" in instance["ctxs"][
            0] else "[ID: {id}] Document: {text}"
        passage_text = "\n\n".join([passage_template.format(**c) for c in instance['ctxs']])

        demos = demo_data.filter(lambda x: x["qid"] != instance["qid"])
        h = abs(int(hashlib.sha256(instance["qid"].encode("utf-8")).hexdigest(), 16) % 2 ** 31)
        demo = demos.shuffle(seed=h)
        demo = drop_duplicates(demo, 'qid').select(range(shots))
        demo_ids = set()
        demo_text = ""
        for d in demo:
            if d["qid"] in demo_ids or len(demo_ids) >= shots:
                continue
            demo_ids.add(d["qid"])
            # sort ids by label
            ids = sorted(d["ctxs"], key=lambda x: x["label"], reverse=True)
            ranking = " > ".join([x['id'] for x in ids])
            demo_text += "\n\n".join([passage_template.format(**c) for c in
                                      d['ctxs']]) + f"\n\nQuery: {d['query']}\nRanking: {ranking}" + "\n\n"
        qrel = [[c['id'], str(c['label'])] for c in instance["ctxs"]]
        input_text = instruction_template.format(demos=demo_text, context=passage_text, question=instance["query"])
        output_text = " > ".join([x['id'] for x in sorted(instance["ctxs"], key=lambda x: x["label"], reverse=True)])
        return {"input": input_text, "output": output_text, "qrel": qrel, "k_values": k_values,
                "query_id": instance["query"], "max_length": max_length}

    instance_data = instance_data.map(_rerank_instance_fillup, remove_columns=["ctxs", "qid", "query"])
    eval_data = instance_data.select(range(eval_questions_num))
    dev_data = instance_data.select(range(eval_questions_num, len(instance_data)))
    return dev_data, eval_data

def load_icl(
        dataset_key: Literal["trec_coarse", "trec_fine", "nlu", "banking77", "clinc150"] = "clinc150",
        max_length: Literal["8k", "16k", "32k", "64k", "128k"] = "8k",
):
    seed = 42
    eval_questions_num = 500  # the 500 questions that will be used in the evaluation set
    shots_map = {
        "trec_coarse": {"128k": 6600, "64k": 3300, "32k": 1600, "16k": 800, "8k": 400},
        "trec_fine": {"128k": 6400, "64k": 3200, "32k": 1600, "16k": 800, "8k": 400},
        "nlu": {"128k": 8296, "64k": 4080, "32k": 2040, "16k": 1020, "8k": 510},
        "banking77": {"128k": 5900, "64k": 2900, "32k": 1450, "16k": 720, "8k": 360},
        "clinc150": {"128k": 7050, "64k": 3525, "32k": 1750, "16k": 880, "8k": 440}
    }
    num_labels_map ={"trec_coarse": 6, "trec_fine": 50, "nlu": 68, "banking77": 77, "clinc150": 151}
    instruction_template = "Use the provided mapping from the text to label to assign a label to the text. Only output \"label: {{label}}\" and nothing else. \n\n{context}\n\n{question}\nlabel:"
    demo_template = "{text}\nlabel: {label}"

    if dataset_key == "trec_coarse":
        all_data = load_dataset("CogComp/trec", trust_remote_code=True)
        text_field, label_field = "text", "coarse_label"
    elif dataset_key == "trec_fine":
        all_data = load_dataset("CogComp/trec", trust_remote_code=True)
        text_field, label_field = "text", "fine_label"
    elif dataset_key == "nlu":
        all_data = load_dataset("xingkunliuxtracta/nlu_evaluation_data", trust_remote_code=True)[
            "train"].train_test_split(test_size=0.1, seed=seed)
        text_field, label_field = "text", "label"
    elif dataset_key == "banking77":
        all_data = load_dataset("PolyAI/banking77", trust_remote_code=True)
        text_field, label_field = "text", "label"
    else:
        all_data = load_dataset("clinc_oos", "plus")
        text_field, label_field = "text", "intent"

    (train_data, test_data) = (all_data["train"], all_data["test"]) if dataset_key != "clinc150" else \
        (all_data["train"], all_data["validation"])

    def _balance_data(data, target_size, seed):
        rand = random.Random(seed)
        label_mapping = {x[label_field]: [] for x in data}
        for x in data:
            label_mapping[x[label_field]].append(x)

        num_rounds = math.ceil(target_size / len(label_mapping))
        new_data = [[] for _ in range(num_rounds)]
        for _, samples in label_mapping.items():
            indices = rand.sample(range(len(samples)), num_rounds % len(samples))
            while len(indices) < num_rounds:
                # sample with replacement if necessary, shouldn't happen unless we have very many shots
                indices += rand.sample(range(len(samples)), min(num_rounds - len(indices), len(samples)))

            for i, idx in enumerate(indices):
                new_data[i].append(samples[idx])

        for i in range(len(new_data)):
            # this shuffles the order of the labels within each set
            rand.shuffle(new_data[i])
        new_data = [item for sublist in new_data for item in sublist][:target_size]
        return new_data

    def _icl_instruction_fillup(instance, demo_data):
        local_seed = (int(hashlib.sha256(instance[text_field].encode("utf-8")).hexdigest(), 16) + seed) % 2**31
        np.random.seed(local_seed)
        demos = _balance_data(demo_data, shots_map[dataset_key][max_length], local_seed)

        label_mapping = list(range(num_labels_map[dataset_key]))
        random.seed(local_seed)
        random.shuffle(label_mapping)

        context = "\n\n".join([
            demo_template.format(text=selected_item[text_field],
                                 label=str(label_mapping[int(selected_item[label_field])]))
            for selected_item in demos]
        )
        input_text = instruction_template.format(context=context, question=instance[text_field])
        output_text = str(label_mapping[int(instance[label_field])])
        return {"input": input_text, "output": output_text, "query_id": instance["text"], "max_length": max_length}

    if dataset_key in ["trec_coarse", "trec_fine"]:
        eval_data = test_data.map(_icl_instruction_fillup, fn_kwargs={"demo_data": train_data}, num_proc=40)
        train_data = train_data.train_test_split(test_size=1000, seed=seed)
        dev_demos_data, dev_data = train_data["train"], train_data["test"]
        dev_data = dev_data.map(_icl_instruction_fillup, fn_kwargs={"demo_data": dev_demos_data}, num_proc=40)
    else:
        test_data = test_data.shuffle(seed=seed)
        eval_data = Dataset.from_list(_balance_data(test_data, eval_questions_num, seed))   # question might duplicate sometimes, but it doesn't matter
        eval_questions = eval_data[text_field]
        dev_data = test_data.filter(lambda x: x[text_field] not in eval_questions, num_proc=40)
        dev_data = dev_data.map(_icl_instruction_fillup, fn_kwargs={"demo_data": train_data}, num_proc=40)
        eval_data = eval_data.map(_icl_instruction_fillup, fn_kwargs={"demo_data": train_data}, num_proc=40)

    return dev_data, eval_data

def _filter_short_seq(data, tokenizer, length=131072, text_key="context"):
    data = data.filter(lambda x: len(tokenizer(x[text_key])["input_ids"]) >= length, num_proc=32)
    return data

def _truncate_context_to_length(instance, tokenizer, length=131072, text_key="context"):
    postfix_text = " ... [the rest of the text is omitted]"
    separator_length = len(tokenizer(postfix_text)["input_ids"])
    tokens = tokenizer(instance[text_key], return_offsets_mapping=True)
    if len(tokens["input_ids"]) > length:
        # truncate the context to the given length, with a pre-defined postfix
        instance[text_key] = instance[text_key][:tokens["offset_mapping"][length-separator_length][1]] + postfix_text
    return instance

def load_long_doc_qa(
        dataset_key: Literal["narrative_qa", "infinite_bench_qa", "infinite_bench_mc"] = "narrative_qa",
        max_length: Literal["8k", "16k", "32k", "64k", "128k"] = "8k",
):
    """Load dataset that belongs to the long document question answering task."""
    shots = 2
    seed = 42
    doc_cut_off_len = 131072
    max_len_map ={
        "narrative_qa": {"128k": 130772, "64k": 65236, "32k": 32468, "16k": 16084, "8k": 7892,},
        "infinite_bench_qa": {"128k": 130862, "64k": 65326, "32k": 32558, "16k": 16174, "8k": 7982,},
        "infinite_bench_mc": {"128k": 130862, "64k": 65326, "32k": 32558, "16k": 16174, "8k": 7982,},
    } # the max len can only be stored in this way, as they are hard-coded from the original code
    max_len = max_len_map[dataset_key][max_length]
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=HF_TOKEN)
    eval_questions_num = 100

    if dataset_key == "narrative_qa":
        instruction_template = "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\n{demos}{context}\n\nQuestion: {question}\nAnswer:"
        all_data = load_dataset("narrativeqa")
        instance_data = all_data["test"].shuffle(seed=seed)
        demo_data = all_data["train"]   # HELMET did not shuffle here, so we keep it the same
        instance_data = instance_data.map(
            lambda x: {"context": x["document"]["text"], "question": x["question"]["text"],
                       "answers": [ans["text"] for ans in x["answers"]]},
            remove_columns=["document"]
        )
        instance_data = _filter_short_seq(instance_data, tokenizer, doc_cut_off_len, "context")
        instance_data = instance_data.map(
            lambda x: {"demos": "For example:\n\n" + "\n\n".join(
                [f"Question: {ex['question']['text']}\nAnswer {ex['answers'][0]['text']}" for ex in
                 demo_data.shuffle().select(
                     range(shots))]) + "\n\nNow, use the following story to answer the question:\n\n"}
        )
        truncate_fn = partial(_truncate_context_to_length, tokenizer=tokenizer, length=max_len, text_key="context")
        instance_data = instance_data.map(truncate_fn, num_proc=16)

        instance_data = instance_data.map(
            lambda x: {"input": instruction_template.format(**x), "output": x["answers"], "query_id": x["question"],
                       "max_length": max_length},
            remove_columns=["question", "answers", "context", "demos"]
        )

        eval_data = instance_data.select(range(eval_questions_num))
        dev_data = instance_data.select(range(eval_questions_num, len(instance_data)))

    else:
        if dataset_key == "infinite_bench_qa":
            instruction_template = "You are given a story and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\n{demos}{context}\n\nQuestion: {question}\nAnswer:"
            demo_template = "[story text]\nQuestion: {input}\nAnswer: {answer[0]}"
            ft = Features({"id": Value("int64"), "context": Value("string"), "input": Value("string"),
                           "answer": Sequence(Value("string")), "options": Sequence(Value("string"))})
            instance_data = load_dataset("xinrongzhang2022/infinitebench", features=ft)["longbook_qa_eng"]
        else:
            instruction_template = "You are given a story and a question with multiple choices. Choose the best answer from the options provided. Only one of the following options is correct, output the answer using one single letter (A, B, C, or D). Don't say anything else.\n\n{demos}{context}\n\nQuestion: {question}\nOptions:\n{options}\nAnswer:"
            demo_template = "[story text]\nQuestion: {input}\nOptions:\n{options}\nAnswer: {answer[0]}"
            ft = Features({"id": Value("int64"), "context": Value("string"), "input": Value("string"),
                           "answer": Sequence(Value("string")), "options": Sequence(Value("string"))})
            instance_data = load_dataset("xinrongzhang2022/infinitebench", features=ft)["longbook_choice_eng"]
            instance_data = instance_data.map(
                lambda x: {
                    "options": "A. {}\nB. {}\nC. {}\nD. {}".format(*x["options"]),
                    "answer": [chr(ord("A") + x["options"].index(x["answer"][0])),
                               f"{chr(ord('A') + x['options'].index(x['answer'][0]))}. {x['answer'][0]}"]
                }
            )


        instance_data = instance_data.map(
            lambda instance: {"question": instance["input"],
                              "demos": "For example:\n\n"
                                       + "\n\n".join([demo_template.format(**ex) for ex in
                                                                         instance_data.filter(
                                                                             lambda x: x["id"] != instance[
                                                                                 "id"]).shuffle(
                                                                             seed=seed).select(
                                                                             range(shots))])
                                       + "\n\nNow, read the following story:"}
        )

        truncate_fn = partial(_truncate_context_to_length, tokenizer=tokenizer, length=max_len, text_key="context")
        instance_data = instance_data.map(truncate_fn, num_proc=16).shuffle(seed=seed)      # HELMET shuffled it again

        eval_data = instance_data.select(range(eval_questions_num))
        dev_data = instance_data.select(range(eval_questions_num, len(instance_data)))
        eval_data = eval_data.map(
            lambda x: {"input": instruction_template.format(**x), "output": x["answer"][0], "query_id": x["question"],
                       "max_length": max_length},    # use the first answer
            remove_columns=["id", "question", "answer", "context", "demos", "options"]
        )
        dev_data = dev_data.map(
            lambda x: {
                "input": instruction_template.format(question=x["question"], context=x["context"], options=x["options"],
                                                     demos="Now, read the following story:"),
                "output": x["answer"][0], "query_id": x["question"], "max_length": max_length},  # use the first answer
            remove_columns=["id", "question", "answer", "context", "demos", "options"]
        )   # the `options` placeholder for infinite_bench_qa will be ignored when constructing the instance here

    return dev_data, eval_data

def load_summarization(
        dataset_key: Literal["infinite_bench_sum", "multi_lex_sum"] = "infinite_bench_sum",
        max_length: Literal["8k", "16k", "32k", "64k", "128k"] = "8k",
):
    """Load dataset that belongs to the summarization task."""
    shots = 2
    seed = 42
    doc_cut_off_len = 65536
    max_len_map = {
        "infinite_bench_sum": {"128k": 129672, "64k": 64136, "32k": 31368, "16k": 14984, "8k": 6792, },
        "multi_lex_sum": {"128k": 130372, "64k": 64836, "32k": 32068, "16k": 15684, "8k": 7492, },
    }  # the max len can only be stored in this way, as they are hard-coded from the original code
    max_len = max_len_map[dataset_key][max_length]
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=HF_TOKEN)

    if dataset_key == "infinite_bench_sum":
        eval_questions_num = 50 # different from HELMET
        instruction_template = "You are given a book and you are tasked to summarize it. Write a summary of about 1000 to 1200 words. Only write about the plot and characters of the story. Do not discuss the themes or background of the book. Do not provide any analysis or commentary.\n\n{demos}{context}\n\nNow summarize the book.\nSummary:"
        ft = Features({"id": Value("int64"), "context": Value("string"), "input": Value("string"),
                       "answer": Sequence(Value("string")), "options": Sequence(Value("string"))})
        instance_data = load_dataset("xinrongzhang2022/infinitebench", features=ft)["longbook_sum_eng"]

        instance_data = instance_data.map(
            lambda instance: {"question": instance["input"],
                              "demos": "For example:\n\n"
                                       + "\n\n".join([f"[story text]\nSummary: {ex['answer'][0].strip()}" for ex in
                                                      instance_data.filter(
                                                          lambda x: x["id"] != instance[
                                                              "id"]).shuffle(
                                                          seed=seed).select(
                                                          range(shots))])
                                       + "\n\nNow, read the following story:"}
        )

        truncate_fn = partial(_truncate_context_to_length, tokenizer=tokenizer, length=max_len, text_key="context")
        instance_data = instance_data.map(truncate_fn, num_proc=16).shuffle(seed=seed)  # HELMET shuffled it again

        eval_data = instance_data.select(range(eval_questions_num))
        dev_data = instance_data.select(range(eval_questions_num, len(instance_data)))
        eval_data = eval_data.map(
            lambda x: {"input": instruction_template.format(**x), "output": x["answer"][0]},
            remove_columns=["id", "question", "answer", "context", "demos", "options"]
        )
        dev_data = dev_data.map(
            lambda x: {
                "input": instruction_template.format(question=x["question"], context=x["context"],
                                                     demos="Now, read the following story:"),
                "output": x["answer"][0], "query_id": x["id"] + "-" + x["question"], "max_length": max_length},
            remove_columns=["id", "question", "answer", "context", "demos", "options"]
        )
    else:
        eval_questions_num = 100
        instruction_template = "You are given the legal documents in a civil rights lawsuit, and you are tasked to summarize the case. Write a concise summary of one paragraph (200 to 250 words). The summary should contain a short description of the background, the parties involved, and the outcomes of the case.\n\n{demos}Legal documents:\n{context}\n\nNow please summarize the case.\nSummary:"
        all_data = load_dataset("allenai/multi_lexsum", name="v20230518", trust_remote_code=True)  # use dataset < 4.0.0
        all_data = all_data.filter(lambda x: x["summary/short"] is not None)

        demo_data = all_data["train"]
        eval_data =  all_data["validation"]
        dev_data = all_data["test"]

        data_update_func = lambda x: {
                "context": '\n\n'.join(x["sources"]),
                "demos": "Example summaries:\n\n" + "\n\n".join(["Summary: {}".format(ex["summary/short"]) for ex in
                                                                 demo_data.shuffle().select(range(shots))])
                         + "\n\nNow, write a summary of the following legal documents.\n",
            }
        eval_data = eval_data.map(data_update_func)
        dev_data = dev_data.map(data_update_func)

        eval_data = _filter_short_seq(eval_data, tokenizer, doc_cut_off_len, "context")
        dev_data = _filter_short_seq(dev_data, tokenizer, doc_cut_off_len, "context")

        truncate_fn = partial(_truncate_context_to_length, tokenizer=tokenizer, length=max_len, text_key="context")
        eval_data = eval_data.map(truncate_fn, num_proc=16).shuffle(seed=seed)
        eval_data = eval_data.select(range(eval_questions_num))
        dev_data = dev_data.map(truncate_fn, num_proc=16).shuffle(seed=seed)

        drop_columns = ["id", "sources", "sources_metadata", "summary/long", "summary/short", "summary/tiny",
                        "case_metadata", "context", "demos"]
        eval_data = eval_data.map(
            lambda x: {"input": instruction_template.format(**x), "output": x["summary/short"],
                       "query_id": x["id"], "max_length": max_length},
            remove_columns=drop_columns
        )
        dev_data = dev_data.map(
            lambda x: {"input": instruction_template.format(**x), "output": x["summary/short"],
                       "query_id": x["id"], "max_length": max_length},
            remove_columns=drop_columns
        )

    return dev_data, eval_data

def load_synthetic(
        dataset_key: Literal["json_kv", "ruler_mk_needle", "ruler_mk_uuid", "ruler_mv"] = "ruler_mk_needle",
        max_length: Literal["8k", "16k", "32k", "64k", "128k"] = "8k",
        dataset_parent_dir: str = DATASET_PARENT_DIR
):
    """
    Load datasets for the synthetic recall task.
    """
    shots = 2   # only used by JSON KV
    seed = 42
    eval_questions_num = 100 if dataset_key == "json_kv" else 40
    file_paths = {
        "json_kv": {"128k": "json_kv/test_k1800_dep6.jsonl", "64k": "json_kv/test_k900_dep6.jsonl",
                    "32k": "json_kv/test_k440_dep6.jsonl", "16k": "json_kv/test_k220_dep6.jsonl",
                    "8k": "json_kv/test_k105_dep6.jsonl", },
        "ruler_mk_needle": {"128k":  "ruler/niah_multikey_2/validation_131072.jsonl", "64k":  "ruler/niah_multikey_2/validation_65536.jsonl",
                            "32k":  "ruler/niah_multikey_2/validation_32768.jsonl", "16k":  "ruler/niah_multikey_2/validation_16384.jsonl",
                            "8k":  "ruler/niah_multikey_2/validation_8192.jsonl",},
        "ruler_mk_uuid": {"128k":  "ruler/niah_multikey_3/validation_131072.jsonl", "64k":  "ruler/niah_multikey_3/validation_65536.jsonl",
                          "32k":  "ruler/niah_multikey_3/validation_32768.jsonl", "16k":  "ruler/niah_multikey_3/validation_16384.jsonl",
                          "8k":  "ruler/niah_multikey_3/validation_8192.jsonl",},
        "ruler_mv": {"128k":  "ruler/niah_multivalue/validation_131072.jsonl", "64k":  "ruler/niah_multivalue/validation_65536.jsonl",
                     "32k":  "ruler/niah_multivalue/validation_32768.jsonl", "16k":  "ruler/niah_multivalue/validation_16384.jsonl",
                     "8k":  "ruler/niah_multivalue/validation_8192.jsonl",},
    }

    data_path = str(Path(dataset_parent_dir) / Path(file_paths[dataset_key][max_length]))
    instance_data = load_dataset("json", data_files=data_path)["train"]

    if dataset_key == "json_kv":
        instruction_template = "{context}\n\nExtract the value corresponding to the specified key in the JSON object below.\n\n{demos}Key: {question}\nCorresponding value:"
        demo_template = "Key: {key}\nCorresponding value:{value}"

        def _json_kv_instruction_fillup(instance):
            demo_text = "\n\n".join(
                [demo_template.format(key=key, value=" " + value) for key, value in instance["demos"][:shots]]) + (
                        "\n\n" if shots > 0 else "")
            input_text = instruction_template.format(context=instance["context"], demos=demo_text, question=instance["question"])
            output_text = instance["answer"]
            return {"input": input_text, "output": output_text, "query_id": instance["question"], "max_length": max_length}

        instance_data = instance_data.map(_json_kv_instruction_fillup,
                                          remove_columns=["context", "demos", "question", "answer"])
        eval_data = instance_data.select(range(eval_questions_num))
        dev_data = instance_data.select(range(eval_questions_num, len(instance_data)))
    else:
        if dataset_key in ["ruler_mk_needle", "ruler_mk_uuid"]:
            instruction_template = "A special magic {type_needle_v} is hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat is the special magic {type_needle_v} for {query} mentioned in the provided text?\nThe special magic {type_needle_v} for {query} mentioned in the provided text is"
        else:
            instruction_template = "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?\nThe special magic {type_needle_v} for {query} mentioned in the provided text are"

        def _ruler_instruction_fillup(instance):
            input_text = instruction_template.format(**instance)
            output_text = instance["outputs"]
            return {"input": input_text, "output": output_text, "query_id": instance["query"], "max_length": max_length}

        instance_data = instance_data.map(_ruler_instruction_fillup,
            remove_columns=["outputs", "context", "length", "index", "answer", "query"]
        ).shuffle(seed=seed)
        eval_data = instance_data.select(range(eval_questions_num))
        dev_data = instance_data.select(range(eval_questions_num, len(instance_data)))

    return dev_data, eval_data
