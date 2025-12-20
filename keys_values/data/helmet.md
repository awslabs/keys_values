# Training & Evaluation Benchmark Derived from HELMET

In this document, we describe the main components of the training and evaluation benchmark. Specifically, we start from briefly introducing the connection between our benchmark and HELMET and explain how our setup differs from the original design. We then delve into the individual task categories and detail how each dataset is processed, including information this is not provided in the original HELMET paper.

All data instances in the benchmark can be viewed as question–answer pairs accompanied by auxiliary context (which may be a single long sequence or a list of shorter contexts, depending on the task). The effective sequence length seen by the model is controlled by manipulating this context. Consequently, HELMET defines five sequence-length scales: 8K, 16K, 32K, 64K, and 128K tokens (with K = 1,024).

However, since HELMET is primarily intended for evaluating different models at inference time, it contains only a limited number of instances per dataset. In addition, its design is not directly suitable for training, as a single question–answer pair may be reused multiple times with varying contexts. To construct training-ready datasets, we adopt the following general procedure (focusing mainly on how we load and structure each dataset):

1. **Instance construction**: We follow the same logic as HELMET for constructing each data instance, including how the context is formed and how the sequence length is controlled.
2. **Unified format**: For each configuration setting, we create a JSON object with at least three fields ["input", "output", "query_id", "max_length"].
3. **Separation of splits**: We explicitly separate instances used by the original HELMET benchmark from the remaining data and export two JSONL files: one intended for downstream training, and one containing only the instances used in the original HELMET tasks.

For clarity, we refer to the newly constructed partition as the development set and to the original HELMET partition as the evaluation set. These two sets are curated to ensure that they do not overlap. In the following, we provide a table summarizing the key metadata of our benchmark.

|Category	|Dataset	|Metrics	|Description	|Size	|
|---	|---	|---	|---	|---	|
|Retrieval-augmented generation	|Natural Questions	|SubEM	|factoid question answering	|(893, 600)	|
|Retrieval-augmented generation	|TrivalQA	|SubEM	|trivia question answering	|(876, 600)	|
|Retrieval-augmented generation	|PopQA	|SubEM	|long-tail entity question answering	|(787, 300)	|
|Retrieval-augmented generation	|HotpotQA	|SubEM	|multi-hop question answering	|(192, 600)	|
|Generation with citations	|ALEC ASQA	|Recall, Cite	|anser ambiguous questions with citations	|(848, 100)	|
|Generation with citations	|ALEC Qampari	|Recall, Cite	|answer factoid questions with citations	|(900, 100)	|
|Passage rerank	|MS MACRO	|NDCG@10	|rerank passage with a query	|(83, 40)	|
|Many-shot in-context leanring	|TREC Coarse	|Accuracy	|question type classification, 6 classes	|(1000, 500)	|
|Many-shot in-context leanring	|TREC Fine	|Accuracy	|question type classification, 50 classes	|(1000, 500)	|
|Many-shot in-context leanring	|NLU	|Accuracy	|task intent classification, 68 classes	|(2094, 500)	|
|Many-shot in-context leanring	|BANKING77	|Accuracy	|banking intent classification, 77 classes	|(2580, 500)	|
|Many-shot in-context leanring	|CLINC150	|Accuracy	|intent classification, 151 classes	|(2600, 500)	|
|Long-document QA	|Narrative QA	|Model-based	|book and movie scirpt question ansering	|(1300, 100)	|
|Long-document QA	|Infinite BENCH QA	|ROUGE F1	|novel QA with entity replacemnt	|(251, 100)	|
|Long-document QA	|Infinite BENCH MC	|Accuracy	|novel multi-choice QA with entity replacement	|(129, 100)	|
|Summarization	|Infinite BENCH Sum	|Model-based	|novel summarization with entity replacement	|(53, 50)	|
|Summarization	|Multi-LexSum	|Model-based	|summarizing multiple legal documents	|(355, 100)	|
|Synthetic recall	|JSON KV	|SubEM	|retrieve a key in json dictionary 	|(500, 100)	|
|Synthetic recall	|RULER MK Needle	|SubEM	|retrieve a number within noisy needles	|(60, 40)	|
|Synthetic recall	|RULER MK UUID	|SubEM	|retrieve a UUID within noisy needles	|(60, 40)	|
|Synthetic recall	|RULER MV	|SubEM	|retrieve multiple values for one needle (key)	|(60, 40)	|

All processing starts from the following endpoint: https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/main/data.tar.gz. After decompression, this archive expands to a directory of approximately 40 GB. The original HELMET benchmark data can also be reconstructed using the code provided in their official repository: https://github.com/princeton-nlp/HELMET.git. In the subsequent sections, we delineate the specific steps used to process these data into our training and evaluation benchmark.

## Retrieval Augmented Generation

Within the RAG category, four popular tasks are included: Natural Questions (NQ), Trivia Question Answering (TriviaQA), Hotpot Question Answering (HotpotQA), and Pop Question Answering (PopQA). These datasets are reformulated as RAG-style tasks, where the context consists of a mixture of golden passages and distracting passages. The data instances for these tasks in the provided files share a common set of fields that largely overlaps with the six fields listed in the next table, with only minor task-specific differences. Notably, the instances used here already deviate from the original formats of the respective source datasets.

|Field	|Description	|Example/Extra note	|
|---	|---	|---	|
|question	|a natural language question	|e.g. "when does season 8 of vampire diaries come out"	|
|answers	|a maximally 20-token response	|e.g. "[on October 21 , 2016, October 21 , 2016]"	|
|positive_ctxs	|a list of passages with answers	|list length typically be 2	|
|hard_negative_ctxs	|a list of hard negative passages with wrong answers	|list length typically be 10	|
|ctxs	|a list of passages	|passages with correct answers randomly inserted, typically 1000	|
|depth	|the complexity of this task	|[0, 1] - the provenance of this score is undocumented, but higher values seem to indicate more challenging examples.	|

For the RAG category, each final data instance is represented in the format ["input", "output", "positive_ctxs", "query_id", "depth", "max_length"]. Downstream users are expected to provide a max_length argument when loading the data, so that the dataset can be constructed from the corresponding (hard-coded) source files. The query_id field can be used to trace the provenance of each question–answer pair.

Each instance in this category is wrapped into the following instruction template:

```
Use the given documents to write a concise and short answer to the question. Write your answer in the following format:
Answer: [answer]

{$DEMOS={Document (Title: {$TITLE}): {$DOC}}\n\nQuestion: {$INPUT}\nAnswer: {$ANSWER}}

{$CONTEXT}

Question: {$QUESTION}
Answer:
```



### NQ & TriviaQA

NQ is a large-scale open-domain question answering dataset constructed from real user queries. TriviaQA consists of question–answer pairs authored by trivia enthusiasts, along with independently collected evidence documents. The NQ and TriviaQA subsets used in this benchmark both follow the description above: each dataset contains 993 question–answer pairs, and each pair has six variants with depth (complexity) values ranging from 0 to 1, specifically {0, 0.2, 0.4, 0.6, 0.8, 1}, for a fixed context length.

For each context-length setting, we reserve 100 question–answer pairs as the evaluation partition, yielding 600 evaluation instances in total. The remaining pairs are used to construct the development partition. For each question–answer pair, we randomly sample a single depth level so that only one variant is included in the final development set, which prevents the model from overfitting by memorizing answers for repeated questions across multiple depth configurations. This results in 893 development instances for NQ and 876 for TriviaQA, where for each query we randomly select a single depth variant of the corresponding question–answer pair.


### HotpotQA

HotpotQA is a large-scale Wikipedia-based question answering dataset with 113K question–answer pairs. The HotpotQA subset we use here differs slightly from the NQ and TriviaQA subsets: each data instance contains only five fields, and the depth field is absent. Nevertheless, in this setup each question–answer pair is still associated with three context variants. In total, there are 987 question–answer pairs.

Across the context-length configurations, we reserve 100 question–answer pairs as the evaluation partition, yielding 300 evaluation instances in total. We then use the remaining 787 pairs to construct the development partition, where for each query we randomly select a single depth variant of the corresponding question–answer pair.


### PopQA

PopQA is a large-scale open-domain question answering dataset consisting of 14K entity-centric question–answer pairs. Each question is generated by applying a template to a knowledge tuple retrieved from Wikidata. The subset we use here contains 292 queries in total with several additional fields; among them, ["subj_id", "prop_id", "obj_id"] are particularly useful for filtering long-tail entities. Each query–answer pair has six variants based on the depth parameter.

For the evaluation partition, we select 100 queries whose subject entities pass a popularity threshold of 3, resulting in 600 evaluation instances (each query contributes six depth variants). We use the remaining 192 queries as the development partition, where for each query we randomly select a single depth variant of the corresponding question–answer pair.



## Generation with citation [Not included for Fine-Tuning]

There are two datasets included in this task category: ALCE_ASQA and ALCE_QAMPARI. This citation task is designed in a quite elegant way: for each question, a set of relevant documents is provided, with up to 2,000 documents indexed in order. Depending on the context-window setting, only a subset of these documents is used. The model is required to include citations to these documents in its response.

Both datasets contain seven fields, as summarized in the following table.

|Field	|Description	|Example/Extra note	|
|---	|---	|---	|
|qa_pairs	|a list of q-a pairs with relevant info	|the schema of each pair ['context', 'question', 'short_answers', 'wikipage']	|
|wikipages	|a list of wikipages	|the schema of each page ['title', 'url']	|
|annotations	|extra annotatons for each q-a	|the schema of each annotation ['knowledge', 'long_answer']	|
|sample_id	|a sample id for used q-a pair	|	|
|question	|a question from qa_pairs	|e.g., *Who has the highest goals in world football?*	|
|docs	|a list of relevant docs to answer the question	|the docs are for the specific question, which contains ['id', 'score', 'text','title'] fields	|
|answer	|an answer with citations	|the longer version of the answer	|

Unfortunately, the datasets do not provide a fully formatted ground-truth target for the cited answers. As a result, although this is a challenging and well-motivated task, we are currently unable to use it directly for fine-tuning. For completeness, for each dataset we return 100 instances as the evaluation partition, and the remaining instances as the development partition, although the latter is not usable at the moment.

For reference, instances in this category are wrapped into the following instruction template:

```
{DEMOS=$INSTRUCTION\nDocument (Title: {$TITLE}): {$TEXT}\n\nQuestion: {$INPUT}\nAnswer: {$ANSWER}}

Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing a document, surround its ID with square brackets, such as [x] to cite document x. To cite multiple documents, simply concatenate the citation markers; for example, use [x][y][z] to cite the documents with ID x, y, and z. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.

Question: {$QUESTION}

{$CONTEXT}

Answer:
```



## Passage Re-rank

The MS MARCO dataset is designed for passage re-ranking, where the goal is to rank relevant passages for a given query. Each instance in the dataset consists of a query and a set of passages retrieved from the web using BM25 over Bing search results. Each passage is annotated as perfect, highly relevant, little relevant, or not relevant.

The subset we use here contains 123 instances (one query per instance). Each instance has three fields: ["qid", "query", "ctxs"], where ctxs is a list of passages and each passage is associated with an "id" and a relevance score "label". The ground truth ranking for each instance is obtained by sorting the passages according to "label" and then "id".

We use the first 40 instances as the evaluation partition, and the remaining 83 instances as the development partition.

For reference, instances in this category are wrapped into the following instruction template:

```
You are provided with a list of documents, each indicated by their ID. Rank each document based on their relevance to the question in descending order from most relelvant to least relevant texts. Include all documents in the rankings. Write your answer using the unique IDs, with the following format:\nRanking: ID3 > ID1 > ID2

{$DEMOS}
{$CONTEXT}

Query: {$QUESTION}
Ranking:"
```




## Many-shot in-context learning

There are five datasets included for the ICL task: TREC Coarse, TREC Fine, NLU, BANKING77, and CLINC150. All of them lie in the question classification / intent classification domain. The main challenge is that the label space is relatively large, with the number of classes ranging from 6 to 151.

Unlike most of the other tasks, where demonstrations are primarily used to guide output formatting, here the demonstrations provide essential information that the model must use to make predictions. Specifically, the demonstrations encode the mapping from semantic labels to ordinal labels, whereas the dataset itself only exposes the ordinal labels. For example, each data instance consists of a question paired with a numerical label, where the number corresponds to an underlying semantic category.

To ensure that each instance fits within the context window, we control the number of demonstrations accordingly. Moreover, because the number of classes is large, the labels used in the demonstrations are chosen to be approximately balanced across classes.

A common strategy is applied across all five datasets: for a fixed number of shots (pre-defined), we randomly sample that many question–answer pairs from the training set. For each instance, the random seed is derived from the text of the instance, ensuring that the selected shots are deterministic for that instance while still differing across instances.

For reference, instances in this category are wrapped into the following instruction template:

```
Use the provided mapping from the text to label to assign a label to the text. Only output \"label: {{LABEL}}\" and nothing else. 

{$CONTEXT={$TEXT}\nlabel: {$LABEL}}

{$QUESTION}

label:
```



### TREC Coarse / Fine

These two datasets are derived from a single repository that provides three fields: ["text", "coarse_label", "fine_label"]. They differ only in label granularity: the coarse setting includes 6 labels, whereas the fine setting includes 50 labels.

The original dataset comes with a train–test split containing 5,425 training instances and 500 test instances. We keep the same construction procedure for the evaluation partition, i.e., shots are sampled from the 5,425 training examples. For the development partition, we subsample 1,000 instances from the training data and draw their shots from the remaining training examples, in order to avoid any data leakage.


### NLU

The source data for the NLU dataset consists of a single training partition with 25,175 instances, each containing the fields ["text", "scenario", "label"]. The label space has 68 classes. We split this partition into 90% training and 10% test. From the test split, we select 500 instances to construct the evaluation partition and use the remaining instances as the development partition. This results in an evaluation set with 500 instances and a development set with 2,094 instances.

### Banking77

The BANKING77 dataset has two partitions, train and test, each with the fields ["text", "label"], containing 10,003 and 3,080 instances, respectively. The label space comprises 77 distinct classes. The test partition is used to construct both the evaluation and development sets. Following the same procedure as above, I select 500 instances for the evaluation set and use the remaining 2,580 instances for the development set.


### Clinc150

The CLINC150 dataset comprises three partitions (train, validation, test) with 15,250, 3,100, and 5,500 instances, respectively. Each instance has two fields, ["text", "intent"], and the label space contains 151 intents. The validation set is used to construct both the evaluation and development sets, resulting in 500 instances for evaluation and 2,600 instances for development.


## Long Document Question Anserwing

There are three datasets included in the benchmark for this task category: NarrativeQA, InfiniteBench QA, and InfiniteBench MC.

An interesting observation from the HELMET paper is that, for most models, performance improves as the context length increases. At first glance, this is somewhat counterintuitive. However, this behavior becomes understandable once we examine how the LongQA task is constructed. In this setup, the context is a long document that contains the answer to the question, and its length often exceeds the model’s maximum context window. When a model is evaluated at a given context-window size, the long document is truncated to fit that window, typically by retaining only the first part of the document. This introduces a problem: if the answer does not appear in the retained portion, the model simply lacks the necessary context to produce a correct response. Nevertheless, we consider it reasonable to include this dataset in our experiments, since this kind of truncation-induced failure is likely to occur frequently in realistic long-context scenarios.

P.S. The length of the source document is measured using a tokenizer, and in our setup we use the meta-llama/Llama-2-7b-hf tokenizer.


### NarrativeQA

The original NarrativeQA dataset contains 10,557 unique instances in its test partition, each with three fields: ["document", "question", "answers"]. Although there are only 1,572 unique documents, each (document, question) combination is treated as a distinct instance. For each question, there is an associated long document that can be used as a reference to produce one or more valid answers.

For our long-context setup, we select only those documents with more than 128K tokens, resulting in 1,330 instances. For a given context-window configuration, each document is truncated to match the specified max_length argument before evaluation. Among these 1,330 instances, we keep the original 100-instance split as the evaluation partition and use the remaining 1,230 instances as the development partition. Demonstration examples are constructed from the question–answer pairs in the train partition of the original NarrativeQA dataset, in order to avoid any data leakage.

For reference purpose, the instance of NarrativeQA will be in the following instruction template --

```
You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible.

{$DEMOS=For example:\n\nQuestion: {$INPUT}\nAnswer {$ANSWER}\n\nNow, use the following story to answer the question}

{$CONTEXT}

Question: {$QUESTION}
Answer:
```

### InfiniteBenchQA

The InfiniteBenchQA dataset contains 351 unique instances, each with five fields: ["id", "context", "input", "answer", "options"]. The answer field is a list of possible answers, although most instances contain only a single answer. The options field is only relevant for the InfiniteBenchMC dataset and is therefore empty in this setting.

For this benchmark, we keep 100 instances as the evaluation partition and use the remaining 251 instances as the development partition. In the original construction, demonstration examples are drawn from other question–answer pairs within the same dataset, with the only constraint that the id differs from the current instance. This is acceptable for inference-time evaluation, but when using the data for fine-tuning it introduces a risk that the model may simply memorize question–answer pairs from previously seen data. Since the demonstrations are primarily intended to enforce output formatting rather than provide additional information, we remove them from the development partition.

For reference, each InfiniteBenchQA instance is wrapped into the following instruction template:

```
You are given a story and a question. Answer the question as concisely as you can, using a single phrase if possible.

{$DEMOS=[$STORY_TEXT]\nQuestion: {$INPUT}\nAnswer: {$ANSWER}\n\nNow, read the following story:}

{$CONTEXT}

Question: {$QUESTION}\nAnswer:
```

### InfiniteBenchMC

There is little difference between the handling of InfiniteBenchMC and InfiniteBenchQA. The main distinction is that the options field is populated in InfiniteBenchMC, containing four candidate answers. We map these options to labels [A, B, C, D], so that the model only needs to output a single letter as its response. This task is therefore very similar to the multiple-choice setup in LongBench-v2. After preprocessing, there are 100 instances in the evaluation partition and 129 instances in the development partition.

For reference, each InfiniteBenchMC instance is wrapped into the following instruction template:

```
You are given a story and a question with multiple choices. Choose the best answer from the options provided. Only one of the following options is correct, output the answer using one single letter (A, B, C, or D). Don't say anything else.

{$DEMOS=For example:\n\n[$STORY_TEXT]\nQuestion: {$INPUT}\nOptions:\n{$OPTIONS}\nAnswer: {$ANSWER}\n\nNow, read the following story:}

{$CONTEXT}

Question: {$QUESTION}
Options:
{$OPTIONS}
Answer:
```



## Summarization

There are two datasets for the summarization task: InfiniteBenchSum and Multi-LexSum. Same as the long document QA task, the provided context will be truncated to fit the context window.

### InfiniteBenchSum

The InfiniteBenchSum dataset follows the same format as its sibling tasks, with five fields: ["id", "context", "input", "answer", "options"]. There are only 103 data instances in total, so we do not adopt the same partitioning strategy as for other datasets. Instead, we use 50 instances as the evaluation partition and the remaining 53 instances as the development partition.

For reference, each InfiniteBenchSum instance is wrapped into the following instruction template:

```
You are given a book and you are tasked to summarize it. Write a summary of about 1000 to 1200 words. Only write about the plot and characters of the story. Do not discuss the themes or background of the book. Do not provide any analysis or commentary.

{$DEMOS=For example:\n\n[$STORY_TEXT]\nSummary: {$INPUT}\n\nNow, read the following story:}

{$CONTEXT}

Now summarize the book.
\nSummary:
```



### Multi-LexSum

The Multi-LexSum dataset contains three partitions: train (2,110 instances), validation (312), and test (616). In our setup, the validation partition is used to construct instances for inference-time evaluation, while demonstration examples are drawn from the train partition. Each original data instance contains seven fields: ["id", "sources", "sources_metadata", "summary/long", "summary/short", "summary/tiny", "case_metadata"], but we only use the sources and summary/short fields. In addition, sources shorter than 65,536 tokens are filtered out.

Thanks to the relatively large dataset size, we keep the 100 evaluation instances exactly as in the original construction and use the (filtered) test partition to construct the development partition, resulting in 355 instances.

For reference, each Multi-LexSum instance is wrapped into the following instruction template:

```
You are given the legal documents in a civil rights lawsuit, and you are tasked to summarize the case. Write a concise summary of one paragraph (200 to 250 words). The summary should contain a short description of the background, the parties involved, and the outcomes of the case.

{$DEMOS=Example summaries:\n\nSummary: {$INPUT}\n\nNow, write a summary of the following legal documents.\n}

Legal documents:
{$CONTEXT}

Now please summarize the case.
Summary:
```



## Synthetic recall

There are four datasets included in this task category: JSON KV, RULER MK Needle, RULER MK UUID, and RULER MV. The main purpose of this task is to extract a certain value from a given input.

### JSON KV

The JSON KV dataset consists of a single training partition with 600 instances. Each instance has the fields
["context", "demos", "question", "answer", "depth", "num_kvs"].
We use 100 examples to construct the evaluation partition and the remaining 500 examples as the development partition.

For reference, each JSON KV instance is wrapped into the following instruction template:

```
{$CONTEXT}

Extract the value corresponding to the specified key in the JSON object below.

{$DEMOS=Key: {$KEY}\nCorresponding value:{$VALUE}}

Key: {$QUESTION}
Corresponding value:
```



### RULER MK Needle / UUID / MV

Each of the three RULER variants (MK Needle, MK UUID, MV) contains 100 examples, with the fields
["index", "input", "outputs", "length", "answer", "type_needle_v", "query", "context"].
Conceptually, one variant requires extracting a single number, another requires extracting a UUID, and the third involves retrieving a set of numbers. We use 40 examples as the evaluation partition and the remaining 60 examples as the development partition for each dataset.

For reference, each RULER instance is wrapped into the following instruction template:

```
A special magic {$TYPE} is hidden within the following text. Make sure to memorize it. I will quiz you about the {$TYPE} afterwards.
{$CONTEXT}
What is the special magic {$TYPE} for {$QUERY} mentioned in the provided text?
The special magic {$TYPE} for {$QUERY} mentioned in the provided text is
```














