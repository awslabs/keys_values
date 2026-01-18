# KeysAndValues: Efficient Language Model Inference, Fine-tuning, and Key-value Caching

This library provides implementations of advanced key-value caching for
efficient long context inference and fine-tuning with large language models.
It sits on top of [LitGPT](https://github.com/Lightning-AI/litgpt/tree/main).

The library is primarily intended for research and evaluation. Using it as part
of a production system will require substantial extra efforts.


## Getting Started

We depend on `LitGPT` and inherits its dependencies. Depending on what you plan
to do, you can:

* Install `LitGPT` via `pip`: In case you do not plan to modify `LitGPT` code.
* Install `LitGPT` from source: In case your project includes modifying `LitGPT`
  as well. If you are not sure, choose this path.

### Install `LitGPT` via `pip`

It is best to create a virtual environment:

```bash
git clone git@github.com:awslabs/keys_values.git
python3 -m venv keyval_venv
. keyval_venv/bin/activate
pip install --upgrade pip
pip install 'litgpt[all,test,extra]'
cd keys_values
pip install -e .
```

Run the tests in order to check whether the installation worked:

```bash
pytest test/
```

### Install `LitGPT` from source

First, install `LitGPT` from source:

```bash
git clone git@github.com:Lightning-AI/litgpt.git
cd litgpt
git checkout main
```

If you plan to modify their code beyond simple changes, it may be better to create
a fork. Next, you need to create a virtual environment:

```bash
python3 -m venv keyval_venv
. keyval_venv/bin/activate
pip install --upgrade pip
cd ${LITGPT_PATH}
pip install -e .[all,test,extra]
cd ${KEYS_VALUES_PATH}
pip install -e .
```

Here, replace `${LITGPT_PATH}` with the source path of `LitGPT` and
`${KEYS_VALUES_PATH}` with the source path of `keys_values`.

Run the tests in order to check whether the installation worked:

```bash
cd ${KEYS_VALUES_PATH}
pytest test/
```


## Example: Long Context Fine-tuning on LongBench V2

This example runs on a single `Nvidia A 100` GPU with 40 GB of RAM.

```bash
cd ${KEYS_VALUES_PATH}
python3 keys_values/__main__.py finetune_long_lora Qwen/Qwen2.5-0.5B --out_dir /home/ubuntu/out/finetune/longcontext_lora --data LongBenchV2 --data.max_seq_length 100000 --data.metadata_dir /home/ubuntu/out/finetune/longcontext_lora/data --head_model seq_classification_on_logits --precision bf16-true --verbose some --kv_cache.name h2o-default --kv_cache.cache_length 16384 --kv_cache.chunk_size 1024 --train.save_interval 10 --train.micro_batch_size 4 --train.global_batch_size 4 --eval.interval 10
```

What is happening here?

* `finetune_long_lora`: Default fine-tuning script for `LoRA`
* `--data LongBenchV2`: Using the `LongBenchV2` benchmark with its data loaders.
  `--data.max_seq_length 100000` filters for sequences less than 100k tokens.
  `--data.metadata_dir` stores metadata information about the dataset, so this
  filtering runs much faster next time.
* `--head_model seq_classification_on_logits` selects head model and loss
  function. The benchmark task is 4-way classification, each class represented
  by a single letter. This loss function reduces the logits to these 4 tokens.
  This is much like asking the model to output a single letter, but only allowing
  for valid class labels.
* `--kv_cache.name h2o-default` selects the KV cache policy (`h2o`) and its
  buffer strategy (`default` -- no quantization). `--kv_cache.cache_length` sets
  the cache length (number of slots). Inference with batches at most this length
  are done exactly with a single forward pass. `--kv_cache.chunk_size` sets the
  chunk size. Sequences are processed in chunks of size
  `cache_length, chunk_size, chunk_size, ...`, the first is called the prefill
  chunk.
* `--train.micro_batch_size` sets the batch size for forward and backward
  computations. `--train.global_batch_size` can be a multiple of the former, in
  which case we use gradient averaging.

If you use an AWS `p4d.24xlarge` instance, you can use 8 A 100 GPUs in parallel.
At present, we support data parallelism via
[Lightning Fabric](https://lightning.ai/docs/fabric/stable/). Modifying the
CLI command above like runs training with an effective batch size of 32:

```bash
cd ${KEYS_VALUES_PATH}
python3 keys_values/__main__.py finetune_long_lora Qwen/Qwen2.5-0.5B --out_dir /home/ubuntu/out/finetune/longcontext_lora --devices 8 --data LongBenchV2 --data.max_seq_length 100000 --data.metadata_dir /home/ubuntu/out/finetune/longcontext_lora/data --head_model seq_classification_on_logits --precision bf16-true --verbose some --kv_cache.name h2o-default --kv_cache.cache_length 16384 --kv_cache.chunk_size 1024 --train.save_interval 10 --train.micro_batch_size 4 --train.global_batch_size 32 --eval.interval 10
```

Here, `--devices 8 --train.micro_batch_size 4 --train.global_batch_size 32` sets the
effective batch size to 32, the per-device batch size to 4, and asks to use 8
devices.

### What's Next?

* Try increasing `kv_cache.cache_length` and `kv_cache.chunk_size`. They have
  the [largest impact on speed and accuracy](#cache-length-and-chunk-size).
* Play around with different [cache policies](#kv-cache-policy-and-configuration),
  or try to use buffer quantization (both by `kv_cache.name`).
* If working on a single GPU, try to use `finetune_offload_lora` instead of
  `finetune_long_lora`, this will free up more memory for the backward pass,
  allowing you to explore options like `grad.layers_per_cell` and
  `grad.chunks_per_cell_multiplier`. Larger values speed up computations, but
  require more GPU memory.
* Your KV cache policy is not supported? Why not implement and
  [contribute it back](#implementing-new-kv-cache-policies) to the community?
* You know how to implement GPU kernels in `CUDA` or `Triton` and would like to
  help speeding up inference and fine-tuning with advanced cache policies?
  Your help would be very welcome! Please [read this](#scaled-dot-product-attention).


## Long Context Inference

The library supports inference in the same rudimentary way than `LitGPT`, but
for contexts of essentially arbitrary length. The code in `generate/base` can
be used in the same way as the original `LitGPT` code.

Having said that, we are aware that this is not competitive with leading
inference libraries, such as [vLLM](https://github.com/vllm-project/vllm) or
[SGLang](https://github.com/sgl-project/sglang). Our library lacks support
for multi-device strategies (context parallelism in particular) as well as
many crucial optimizations.

We are actively working towards supporting multi-device fine-tuning in a better
way than what we currently have. As for inference, neither vLLM nor SGLang
support advanced selective KV cache policies in more than an adhoc fashion. If
you want long contexts, you need to provide many GPUs (and cannot use them to
increase batch size). A good strategy would be to try and integrate our KV cache
abstractions and basic implementations there, but rely on their advanced scaled
dot product attention (SDPA) kernels and multi-device low level code.

If you are motivated to work on such an integration, please do get in touch
(see [CONTRIBUTING.md](./CONTRIBUTING.md))! We would love to support users
being able to run inference with long contexts without having to spend a lot
of money on many GPUs, and we think that advanced selective KV cache policies
are an important factor for achieving this goal.

A script for evaluating fine-tuned models on long context test data is provided
in [finetune/longcontext_eval.py](./keys_values/finetune/longcontext_eval.py).


## Long Context Fine-tuning

A major distinguishing factor of this library is its support of long context
fine-tuning. Importantly, we fine-tune a model with a particular KV cache
policy in place. Existing solutions for long context fine-tuning either
restrict the model to a different architecture or store the key-value information
exactly, distributed across several GPU devices (this is called *context
parallelism* or *RingAttention*).

Context parallelism is a good choice if you have the required GPUs (you cannot
use them to achieve larger batch size then), and if you also require exact KV
caching across multiple GPU at inference time. However, if you like to use
advanced selective KV caching during inference (such as H2O), maybe on a single
device only, it may not be a good idea to use context parallelism for fine-tuning,
because this is not aware of the cache restrictions put in place during
inference. In contrast, the techniques provided here compute gradients with your
KV cache policy in place, which allows the model to adapt to it.

The following fine-tuning modes are currently provided:

* [finetune_long_lora](./keys_values/finetune/longcontext_lora.py): Fine-tune
  parameters of LoRA adapters. Supports distributed data parallelism.
* [finetune_long_full](./keys_values/finetune/longcontext_full.py): Fine-tune
  all model parameters. Supports distributed data parallelism. This is not a
  good choice with `Adam` optimization, because the optimizer state is too large
  to fit into GPU memory (this is independent of context lengths). Unfortunately,
  our gradient computation clashes with assumptions made in `PyTorch
  distributed`, so you cannot easily use fully sharded data parallel.
* [finetune_offload_lora](./keys_values/finetune/longcon_offload_lora.py):
  Fine-tune parameters of LoRA adapters, using CPU offloading. We keep model
  weights and optimizer state on the CPU, running forward and backward on
  copies on the GPU. The backward pass uses model shards, which frees up GPU
  memory which can be used to speed up computations. This is the best choice
  for exploring our method on single GPUs with 40 GB of RAM or less. Distributed
  data parallel is not supported at the moment, but this is high on our agenda.
* [finetune_offload_full](./keys_values/finetune/longcon_offload_full.py):
  Fine-tune all model parameters, using CPU offloading. Use this to explore
  full weights fine-tuning with `Adam optimizers`.

They mostly share the same command line arguments, which are detailed in the
sequel.

### Basic Arguments

The scripts are called as follows:

```bash
python3 keys_values/__main__.py {mode} {model} [{command line args}]
```

Here, `mode` is the fine-tuning mode (`finetune_long_lora`, `finetune_long_full`,
`finetune_offload_lora`, `finetune_offload_full`), and `model` is the Hugging Face model name (for example,
`Qwen/Qwen2.5-0.5B` selects the 0.5B parameter version of Qwen 2.5). You can also
put a checkpoint path here. The Hugging Face model must be supported by `LitGPT`,
the default configuration is taken from there.

Basic arguments are:

* `precision`: Precision to be used for weights. The same is used for KV cache
  buffers.
* `devices`: Not for `finetune_offload_*` modes. Number of GPU devices to be
  used. Defaults to 1.
* `device`: For `finetune_offload_*` modes only. Number of GPU device on which
  computations are done. Defaults to 0.  The model and optimizer state are kept
  on the CPU.
* `verbose`: Verbosity level, can be "none", "some", "more", "all".
* `train.*`: Parameters controlling training. This is taken from `LitGPT` without
  modification. Most important ones:
  - `train.micro_batch_size`: Batch size for individual computations on single
    device.
  - `train.global_batch_size`: Not for `finetune_offload_*`. Batch size used
    for optimizer updates. Must be  multiple of `train.micro_batch_size`. If
    `train.global_batch_size == train.micro_batch_size * devices`, this is
    distributed data parallel.
  - `train.save_interval`: Number of optimizer steps between saving checkpoints.
* `eval.*`: Parameters controlling evaluations on validation set. Taken from
  `LitGPT` with little modification. Most important ones:
  - `eval.interval`: Number of optimizer steps between evaluations.
  - `eval.initial_validation`: Run validation before training starts? If this
    is `False`, we run validation on two cases just to check whether things
    break.
  - `eval.final_validation`: Run validation after end of training?
  - `eval.micro_batch_size`: Local batch size to be bused for validation. Overrides
    `train.micro_batch_size`. This can often be larger, because evaluation needs
    less GPU memory than training.
* `lora.*`: Only for `finetune_long_lora`, `finetune_offload_lora` modes.
  Controls LoRA parameterization of  base model. This is taken from `LitGPT`
  without modification. Most important ones:
  - `lora.r`: Rank of LoRA parameterization. One axis of LoRA parameters have
    this size.
  - `lora.alpha`: This parameter is needed for scaling updates as `alpha / r`.
    "This scaling helps to reduce the need to retune hyperparameters when we
    vary r", see [Section 4.1](https://arxiv.org/pdf/2106.09685.pdf).
  - `lora.dropout`: Dropout applied to input in the LoRA branch (before
    multiplying with matrix `A`)
  - `lora.query`: Apply LoRA to linear map to `query`?
  - `lora.key`: Apply LoRA to linear map to `key`?
  - `lora.value`: Apply LoRA to linear map to `value`?
  - `lora.projection`: Apply LoRA to linear projection at end of multi-head
    self attention?
  - `lora.mlp`: Apply LoRA to linear maps of feed-forward network?
  - `lora.head`: Apply LoRA to linear map to logits in the head?

### Dataset and Loss Function

These arguments select the dataset for training and evaluation, as well as the
loss function and head model to be used. We inherit dataset management from
`LitGPT`, in that a subclass of `litgpt.data.DataModule` needs to be provided.
An example is given by [data.LongBenchV2](./keys_values/data/longbench_v2.py#L127).
All `DataModule` subclasses imported in the script file can be chosen by `--data`.
Moreover, `--data.*` is used to set constructor parameters for the dataset.

Relevant arguments for `LongBenchV2` (which is the default dataset):

* `data.max_seq_length`: If given, we filter sequences to have token length
  less or equal this limit. The remaining data is split into training and
  validation sets.
* `data.metadata_dir`: If given, we store meta data into this directory. In
  particular, we tokenize all sequences and determine their token lengths, so
  that filtering runs much faster in the next call, independent of the value
  of `data.max_seq_length`.
* `data.val_split_fraction`: The fraction of the dataset to use for the
  validation dataset. The rest is used for training.
* `data.trainloader_longest_first`: If `True`, the training dataloader returns
  the longest sequences in the first batch. This is useful in order to detect
  out of memory errors early.<br>
  Note that the current implementation does not otherwise cluster the cases so
  that sequences in a batch are most similar in length.
* `test_set_tag`: If this is given, we also maintain a test dataset and serve a
  test dataloader. The tag determines how the test set is chosen. Current
  choices:
  - "rest": All cases with sequence length > `data.max_seq_length`, sorted by
    token sequence length (non-decreasing).

Training loss function and head model are represented by
[HeadModel](./keys_values/head_model.py#L24). In general, the LLM outputs a logits
tensor over the vocabulary, which the head model maps to a loss function value,
given a targets tensor as well. Head models support chunk-wise evaluation in
order to limit the amount of memory needed. The main method is

```python
def forward(
    self,
    model_outputs: torch.Tensor,
    targets: Optional[torch.Tensor],
    input_pos: int,
) -> torch.Tensor:
```

* `model_outputs`: `(batch_size, chunk_size, config.padded_vocab_size)` or
  `(batch_size, chunk_size, config.n_embd)`. Outputs of the LLM for input
  batch of shape `(batch_size, chunk_size)`.
* `targets`: `(batch_size, target_size)` or `None`, where
  `target_size <= chunk_size`. If shorter, they align with `model_outputs`
  on the right. If `None`, the model outputs are processed only (part of
  input prompt).
* `input_pos`: Position in total sequence. Starts with `input_pos=0`. Must
  be increased by `chunk_size` afterwards. This is not done by the `HeadModel`.

This is called sequentially over chunks, from left to right, and `input_pos=0`
starts a new batch. While most loss functions are just additive, some have a
state which allows for other aggregation modes over chunks. For some loss
functions, `targets` is passed with the final chunk only. If a loss function
is normalized over the number of targets, the
[HeadModel.num_target_entries](./keys_values/head_model.py#L73) method is used
in order to determine the normalization constants for each part.

For head models which operate on top of logits outputs, the
[HeadModel.needs_logits](./keys_values/head_model.py#L35) method returns `True`.
If this returns `False`, the head model operates on top of final layer outputs,
so the LLM skips the final linear map to logits.

The following head models are currently supported:

* `--head_model next_token_prediction`:
  [CrossEntropyOnLogits](./keys_values/head_model.py#L132). Cross-entropy loss
  on target tokens. Needs logits. `targets` can be shorter than `model_outputs`,
  in which case they are aligned on the right. The current implementation only
  supports this specific type of masking.<br>
  For next-token prediction, ensure that the inputs to the LLM and the targets
  are based on the same sequences, but shifted by one token position.
* `--head_model seq_classification_on_logits`:
  [SequenceClassificationOnLogits](./keys_values/head_model.py#L222). Works for
  multi-way classification. Needs logits. The label of each class must be
  represented by a single token. The logits output by the LLM are restricted to
  the class label tokens, then cross-entropy loss is applied. For example,
  `LongBenchV2` is 4-way classification with class labels `A`, `B`, `C`, `D`.
  The logits for these 4 tokens are selected and fed into the cross-entropy
  loss.<br>
  `targets.shape[1] == 1` for the last chunk (single token), `targets=None` for
  the other chunks. This is simpler for the model to learn than using
  `--head_model next_token_prediction` with classification targets, because
  the model cannot output anything different from class labels.
* `--head_model seq_classification`:
  [SequenceClassification](./keys_values/head_model.py#L310). Works for
  multi-way classification. Does not need logits. Here, the head model
  contains a linear map from last layer outputs to logits over class labels,
  whose weights are fine-tuned alongside LLM weights (in return, the final
  linear map in the LLM is not trained). For example, `LongBenchV2` is 4-way
  classification with class labels `A`, `B`, `C`, `D`,  the linear map in the
  head model is given by `torch.nn.Linear(config.n_embd, 4, bias=True)`.

### KV Cache Policy and Configuration

For more details on our KV cache abstractions, please study the docstrings in
the codebase. We are preparing a comprehensive technical report on all novelties
implemented here.

A KV cache can be thought of being represented by these variables:
```python
{
    "keys": torch.Tensor(batch_size, n_query_groups, cache_length, head_size),
    "values": torch.Tensor(batch_size, n_query_groups, cache_length, head_size),
    "token_pos": torch.Tensor(batch_size, n_query_groups, cache_length),
}
```

It has up to `cache_length` slots, where key-value information can be stored.
Each slot provides an array of shape `(batch_size, n_query_groups, head_size)`,
in that every batch dimension and query group has its own key and value vectors.
We cannot say that a token (position) is in the cache or not: it may be in the
cache for some `(b, h)`, but not for others. Also, `token_pos[b, h, j]` is the
token position (in the complete sequence batch) for which `keys[b, h, j, :]`,
`values[b, h, j, :]` stores KV information. This is important for book-keeping,
but also to create the causal attention masks for multi-head self attention.
In other words, we do not maintain keys and values as block-sparse tensors, but
as standard dense tensors: this is simple and allows us to use normal `PyTorch`
operators. `token_pos` matters only when creating attention masks. Moreover,
we use `torch.gather` to extract information for slots, and `torch.scatter`
to write information for new tokens into the cache.

For the CLI, a cache is identified by `kv_cache.name`, which can be a string
`{cname}-{bname}`, where `cname` determines the KV cache policy (i.e., which
slots are overwritten once the cache is full) and `bname` determines the buffer
strategy (i.e., how is the KV information stored). These KV cache policies are
currently supported:

* `dense`: [DenseKVCache](./keys_values/kvcache/basics.py#L296). Represents
  exact KV caching, in that the KV information for all tokens is stored. Can
  only be used for sequences of length up to `cache_length`.
* `lastrec`: [LastRecentlyInsertedKVCache](./keys_values/kvcache/basics.py#L478).
  This cache maintains KV information for the `cache_length` last recently
  inserted tokens in the cache. When the cache is full, new information
  overwrites slots which have not been overwritten for the longest time.
* `h2o`: [H2OKVCache](./keys_values/kvcache/h2o.py#L28). Implements an improved
  variant of the heavy hitter oracle (H2O) strategy (for citation, see
  docstring). H2O scores each `(b, h, j)` by the sum of attention weights
  assigned to the KV pair since it is in the cache. Information is evicted if
  this "usage" score is lowest. In a strong sense, H2O implements the least
  recently used (LRU) strategy known from general caches. It requires scaled
  dot product attention (SDPA) to return summed attention weights.<br>
  We implement a number of simple improvements over what has been published as
  H2O.
* `qh2o`: [QuantizedH2OKVCache](./keys_values/kvcache/qh2o.py#L31).
  When H2O is combined with buffer quantization (which is recommended), it can
  be improved by taking quantization errors into account, as has been published
  in a follow-up paper (see docstring for citation).
* `h2o-vlen`: [VLengthH2OKVCache](/keys_values/kvcache/h2o.py#L334). Replaces the
  H2O cumulative attention weights score with an expected value norm score,
  which accounts for the length of value vectors as well. In the end, the
  attention output is a linear combination of value vectors, so these lengths
  should play a role. Can be used as alternative to `h2o`.
* `qh2o-vlen`: [QuantizedVLengthH2OKVCache](./keys_values/kvcache/qh2o.py#L216).
  Combination of `h2o-vlen` and `qh2o`. Can be used as alternative to `qh2o`.
* `h2o-orig`: [H2OOriginalKVCache](./keys_values/kvcache/h2o.py#L482). Implements
  the H2O cache policy as originally published. This has some shortcomings which
  we corrected with `h2o`. This cache is for comparison purposes only, we do not
  recommend to use it otherwise, use `h2o` or the other variants instead.

The KV cache information across all layers of a model often takes more space on
the GPU than the model weights. It therefore makes sense to compress KV
information by quantization (compression and decompression must be very fast).
This is directed by the buffer strategy, which can be combined the KV cache
policy. Note that KV information is maintained with the same `dtype` as model
weigths, so typically `float16` or `bfloat16`. Buffer strategies are:

* `default`: [DefaultKVCacheBuffers](./keys_values/kvcache/buffers.py#L390).
  Buffers are stored as is, no compression. This is fastest, but needs the most
  GPU memory.
* `torch-quantized8`:
  [TorchBasicQuantizer](./keys_values/kvcache/quantize/pytorch.py#L119). Default
  `PyTorch` quantization to 8 bits. This quantizer works on CPU as well.
* `ao-quantized4`, `ao-quantized8`:
  [TorchAOQuantizer](./keys_values/kvcache/quantize/torch_ao.py#L27). `torchao`
  quantization to 4 or 8 bits. GPU only.
* `bnb-quantized4`, `bnb-quantized8`:
  [BitsAndBytesQuantizer](./keys_values/kvcache/quantize/bitsandbytes.py#L48).
  `bitsandbytes` quantization to 4 or 8 bits. GPU only.

With 16 bit standard `dtype`, 4 bit quantization reduces GPU memory requirements
by a factor of 4, allowing you to choose a larger `cache_length`.

The most important parameters for KV caching are `kv_cache.cache_length` and
`kv_cache.chunk_size`, they are discussed [below](#cache-length-and-chunk-size).
Other important arguments can be specified as `kv_cache.cache_kwargs.*`. They
are:

* `grace_period`: Not for `dense`, `lastrec`. For a score-based cache policy, we
  can define a grace period. Tokens which enter the cache at position `t` cannot
  be evicted before position `t + grace_period` then. A grace period makes sense
  if scores are noisy when tokens are in the cache for a short time only.
* `max_chunk_size`: Not for `dense`, `lastrec`. Limits the length
  `query.shape[2]` for calls to `kv_cache.forward` except for the prefill (when
  `input_pos == 0`). This is used to speed up finding the score minimizers.
* `keep_initial_fraction`: Not for `dense`, `lastrec`. See docstring of
  [AttnWeightsKVCache](./keys_values/kvcache/attn_weights.py#L283).
* `normalize_scores`: Not for `dense`, `lastrec`. Scores are cumulative over
  the time (in token positions) some entry is in the cache already. This may
  favor earlier tokens. Scores are normalized by the age of the entry if
  `normalize_scores=True`.

### Cache Length and Chunk Size

The most important argument for a KV cache is `kv_cache.cache_length`, the
number of slots. Sequences with no more than this number of tokens are processed
with a single forward pass and no cache evictions. Also, the first *prefill*
chunk to be processed is typically of this size, while subsequent chunks (if
any) are smaller.

**Note**: Our code supports different KV cache lengths for each layer, but this
is not yet enabled for the CLI.

As a rule of thumb, choose the cache length as large as possible, before you
run out of memory. Run inference with the longest batch first, using
`--data.trainloader_longest_first True`.

The next most important parameter is `kv_cache.chunk_size`. This is not a property of
the cache (except see `max_chunk_size`), but of inference and gradient
computation. We process a batch of long sequences in chunks. The first chunk
has length close to `cache_length`, subsequent chunks are shorter,
typically of length `chunk_size`. The larger the chunk size is, the faster a
long sequence (prompt) can be processed, but there is an important catch. Once
a KV cache is full, new KV information overwrites earlier content. This is done
in chunks of `chunk_size`. Here, the larger the chunk size, the worse the
approximation to exact KV caching becomes. As an extreme case, if
`chunk_size = cache_length`, the KV cache policy is not used at all, and
inference behaves as if the sequence was split into `cache_length`-sized
chunks, which are processed independently from each other!

This means that `chunk_size` is a real hyper-parameter, which determines both
runtime, but also approximation accuracy, which can affect overall accuracy.
Note that GPU memory requirements do not strongly depend on `chunk_size`.

Finally, if `--kv_cache.randomize_chunk_sizes True` is used, then chunk sizes
after the first are picked at random from a distribution with mean
`kv_cache.chunk_size`. The idea behind randomized chunk sizes is to ensure the
model does not adapt to a fixed chunk size.

### Optimizer

The most popular stochastic gradient optimizers from `PyTorch` can be selected,
and others can easily be added. Optimizer arguments are:

* `--optimizer {name}`: Choose among
  [SUPPORTED_OPTIMIZERS](./keys_values/finetune/args.py#L167). Defaults to
  "AdamW".
* `optimizer.learning_rate`: Base learning rate
* `optimizer.weight_decay`: Weight decay constant
* `optimizer.eps`: Eps constant
* `optimizer.momentum`: Momentum constant (if supported)
* `optimizer.dampening`: Dampening constant as part of momentum (if supported)
* `optimizer.adam_betas`: Only for `Adam` optimizers. Tuple `(beta1, beta2)`
* `optimizer.adadelta_rho`: Only for `Adadelta`
* `optimizer.rmspprop_alpha`: Only for `RMSprop`

### Multi-head Self Attention, Scaled Dot Product Attention

Key-value information supports the computation of multi-head self attention (MHA),
in the case when queries are shorter than (and aligned on the right with) keys
and values. For token generation, `query` has length 1, while for processing
a long prompt, it often has length close to `chunk_size`. In fact, our KV
cache abstraction has [KVCache.forward](./keys_values/kvcache/base.py#L197)
computing in this case, when `query`, `key`, `value` correspond to *new tokens*.
For exact KV caching, `key` and `value` would be appended to the existing
buffers. In general, they overwrite slots in the cache buffers, evicting the
information for earlier tokens if the cache is full.

The typical structure of this `forward` call is implemented in
[DefaultKVCache.forward](./keys_values/kvcache/base.py#L520). After the cache
is updated, we make a `self.mha(...)` call, passing `query` along with the
full cache content for keys and values. This
[MultiHeadSelfAttention](./keys_values/attention.py#L95) abstraction computes
the *scaled dot product attention* (SDPA) inner part of MHA, after `query,
key, value` are determined and position encoded. SDPA is by far the
computationally most crucial primitive in LLM inference and is usually
represented by highly optimized SDPA kernels written in CUDA.

#### Position Encoding, YaRN

We implement `RoPE` for position encoding, essentially following `LitGPT`. In
terms of adjusting `RoPE` for sequence length, we use `YaRN`, see docstring
of [YaRNPositionEncoding](./keys_values/pos_encoding.py#L259). This can be
switched off with `--yarn_rope False`, in which case the same static RoPE
is used for all sequences. This is not recommended.

Note that KV information passed to SDPA and stored in KV caches has keys (and
queries) encoded already. This works for fine-tuning and inference with some
expected sequence length. Dynamic YaRN would adjust RoPE during inference,
this is not implemented yet. For such a use case, KV information would have to
be stored before encoding.

#### Scaled Dot Product Attention

Scaled dot product attention (SDPA) is represented by
[MultiHeadSelfAttention.__call__](./keys_values/attention.py#L209). Ideally, its
implementations are via fast kernels, such as
[torch.nn.functional.scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
However, we have some special requirements:

* Some KV cache policies require attention weights on top of attention outputs
  returned by SDPA. The full attention weights would be a tensor of shape
  `(batch_size, n_head, q_len, kv_len)`, where `q_len = query.shape[2]`,
  `kv_len = key.shape[2]`, which is much too big to maintain in memory. We
  ask for attention weights summed over the query axis, shape
  `(batch_size, n_head, kv_len)`, with `return_attn_weights=True`. This is
  sufficient to compute H2O and other scores.
* We need the "rectangular" case, where `1 << q_len << kv_len`, not just the
  "training" (or prefill) case, `q_len == kv_len`, which most SDPA kernel
  developers focus on almost exclusively.
* We need implicit causal attention masking even if `key`, `value` are
  reordered, as expressed by `kv_cache.token_positions`. This is the least
  important requirements, since `key`, `value` can cheaply be reordered.

We are currently working actively to improve the SDPA kernel situation for this
library (and would be very happy for help, see
[CONTRIBUTING.md](./CONTRIBUTING.md)). At present, we support two kernels:

* Query-padded PyTorch SDPA: We use
  `torch.nn.functional.scaled_dot_product_attention`, but pad `query` with
  zeroes on the left to obtain the square "training" case. We also reorder
  `key`, `query` so that the new entries (corresponding to `query`) are on
  the right end. Cannot return attention weights.
* Naive blockwise SDPA: We use an own implementation
  [scaled_dot_product_attention_in_blocks](./keys_values/attention.py#L477).
  The computation is done in blocks so that no more than `tmp_array_limit_gb`
  GB of GPU memory is needed for the temporary buffers.

We ran an experiment for many different `kv_len` to determine from which
`q_len` value onwards query-padded SDPA is faster. However, if attention
weights are required, we currently have to use naive SDPA even for large
`q_len`.

Note that SDPA for the initial prefill call always uses the fast PyTorch SDPA.
This is because no scores are computed then, and so attention weights are not
needed even for H2O policies.

Relevant arguments are:

* `attention_forward_temp_size_gb`: Size limit (in GB) for temporary buffers
  in naive SDPA, used in `forward` pass.
* `attention_backward_temp_size_gb`: Same size limit, but for SDPA computations
  during the `backward` pass. This is discussed [below](#gradient-computation).

### Gradient Computation

For more details on how gradient are computed in the presence of KV caches
(this is a novel contribution of this library), please study the docstrings in
the codebase. We are preparing a comprehensive technical report on all novelties
implemented here.

The main difficulty of computing gradients for long context models is large
GPU memory requirements. Even if gradients are blocked for KV cache score
computations, just using `torch.autograd` is out of the question. We do not
go into full details, but our technique is a combination of several ideas:

* Splitting backward computations into cells: Think of computations as an
  array, the vertical axis being the model layers, the horizontal axis being
  the sequence chunks. The first column has entries of length close to
  `cache_length`, remaining columns have length `chunk_size`. We tile this
  array with cells. A row of cells covers up to `grad.layers_per_cell` layers,
  a column of cells covers a number of chunks.
* Activation and KV cache checkpointing: We run `torch.autograd` gradient
  computation on each cell. This needs inputs and head gradients for each cell.
  Inputs are obtained by activation checkpointing during forward pass
  (horizontal) and checkpointing KV cache buffers (vertical). Checkpoints are
  stored on CPU, possibly quantized. Since KV cache buffers are much larger,
  we only checkpoint them for the current row of cells.

To be precise, gradients are computed in two phases:

* Forward phase: This is what we also do for inference, with KV cache policies
  in action. However, we store activation checkpoints at each cell boundary
  to CPU, and we also log all KV cache eviction decisions into a so-called
  *replay log*.
* Backward phase: In this phase, we use *replay caches*. These are replicas of
  the original KV caches, but instead of running a policy depending on inputs,
  they just replay all decisions made during the forward pass. The backward
  phase moves top down over rows of cells. For each row, we first run
  forward over chunks to store KV cache checkpoints on CPU. Then, we loop
  backwards over cells, running `torch.autograd` to accumulate gradients.

Two more ideas are important. The larger cells are the faster our method runs,
because `torch.autograd` is best run as few times as possible on larger graphs.
However, `autograd` stores tensors in its compute graph which are needed during
the backward pass, which quickly fills up GPU memory. The largest such nodes
are KV cache buffers `keys`, `values` after each cache update, of size
`(batch_size, n_query_groups, cache_length, head_size)`. However, a single
chunk update of them is represented by `torch.scatter` calls with *new* entries
of size `(batch_size, n_query_groups, chunk_size, head_size)`. It is not hard
to see that we can reconstruct the sequence of cache buffers per chunk in the
backward direction, storing nodes of the latter size in the `autograd` graph
only.

Implementing this simple idea in `PyTorch` ends up quite challenging, see
[CellComputationAutogradHooks](./keys_values/kvcache/gradient/autograd_hooks.py#L382).
We use the [autograd saved tensors hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)
mechanism. This has some shortcomings, which renders our code somewhat complex.
However, it is only with this mechanism that we can run our method with
non-trivial cell sizes (i.e., not one cell per layer and chunk). How large
should a cell be in the horizontal direction? We argue that the sum of chunk
lengths for a cell should be approximately `cache_length`. With this convention,
the size of tensors stored in the `autograd` graph scales with `cache_length`
rather than `chunk_size`, so becomes comparable to KV cache size.

Second, when using `torch.nn.functional.scaled_dot_product_attention` as
operator, we find that this creates several large arrays in the `autograd` graph.
To get around this, we implemented our own `PyTorch` operator
[KVCacheScatterUpdateAndSDPAFunction](./keys_values/kvcache/gradient/sdpa_op.py#L474).
for SDPA fused with `torch.scatter` KV cache update. Its `backward` requires naive
blockwise SDPA. We are working on a CUDA version for this fused SDPA operator,
which will speed up computations without sacrificing memory efficiency (like
PyTorch SDPA does).

Important arguments for gradient computations are:

* `--grad.layers_per_cell`: Second phase GPU memory requirements depend
  linearly on this number. It states how many layers are processed in a cell.
  The default is 1. Larger values mean less sequential processing, so faster
  computation. Note that the CPU memory for layer input checkpoints scales
  inverse linearly with this number.
* `--grad.chunks_per_cell_multiplier`: The length of a cell is the sum of
  its chunk's lengths. If `max_cell_length = int(kv_cache.cache_length *
  grad.chunks_per_cell_multiplier)`, chunks are grouped into a cell until
  its length is close to `max_cell_length`, but not larger. By default,
  `grad.chunks_per_cell_multiplier = 1`, so cells are about as long as
  the KV cache. For larger values of the multiplier, there are fewer cells per
  row, which speeds up computations. Second phase GPU memory requirements
  depend linearly on this number.

These two are important hyper-parameters, to be adjusted to use as much of
the available GPU as possible. Further arguments are documented in
[GradientArgs](./keys_values/finetune/args.py#L112), use them as `grad.*`.

How to choose `layers_per_cell` and `chunks_per_cell_multiplier`? They determine
GPU memory usage during the *second phase* only. Their choices becomes most
relevant when CPU offloading of the weights is used as well (so
`finetune_offload_lora`, `finetune_offload_full` modes). In that case, we free
up GPU memory specifically during the second phase by keeping only model weights
for the layers in the current row of cells in GPU memory: all this memory should
be used to increase cell size, which speeds up computations. We can increase cell
width by `chunks_per_cell_multiplier`, cell height by `layers_per_cell`. The
trade-off is:

* Maximize `chunks_per_cell_multiplier`, keep `layers_per_cell=1`: Most weights
  are offloaded, leaving most GPU memory for cells. Fewer cells run faster, less
  GPU memory for KV cache checkpoints.But more activation checkpoints are written
  and read, which is slower due to GPU-CPU synchronization.
* Maximize `layers_per_cell`, keep `chunks_per_cell_multiplier=1` (or even
  below): More weights are kept on GPU in `backward`, and there are more KV cache
  checkpoints, but less activation checkpoints are written and read.


From this
perspective, it may be advantageous to keep `layers_per_cell=1` and maximize
`chunks_per_cell_multiplier`, since most weights are offloaded then. On the other
hand, this requires more activation checkpoints to be written, which can be
slower.

### Profiling GPU Memory and Runtime

This is based on https://pytorch.org/blog/understanding-gpu-memory-1/. It shows
how to profile GPU memory usage during certain parts of forward and backward pass.

GPU memory profiling is activated with the CL argument `--record_gpu_memory_snapshots 100000`.
The number is the `max_entries` argument for `torch.cuda.memory._record_memory_history`.
The kind of profiling is chosen with `--record_gpu_memory_kind`, with values 0, 1,
2, 3. All of them write pickle files to `${OUT_DIR}/gpu_memory_snapshots/`. For
`record_gpu_memory_kind=0`:

* `${OUT_DIR}/gpu_memory_snapshots/iteration${ITER}/snapshot_initial.pickle`:
  From start of iteration until backward over top-most layer. Includes the
  forward pass for layer input checkpoints and KV cache logs, as well as the
  backward for the head model.
* `${OUT_DIR}/gpu_memory_snapshots/iteration${ITER}/snapshot_layer${FST_LAYER_IDX}.pickle`:
  Backward over one row of cells. Here, `FST_LAYER_IDX` the index of the first
  layer for the row of cells.

Here, `OUT_DIR` is given by the CL option `--out_dir ${OUT_DIR}`, `ITER` is the
iteration number. Copy the snapshot files from the GPU instance.

To watch a snapshot, you can try to upload the pickle file to their web
interface at https://docs.pytorch.org/memory_viz. If this does not work for you,
you need a script from `PyTorch`. Clone the `PyTorch` sources to `PYTORCH_PATH`, then
run:
```bash
python3 ${PYTORCH_PATH}/torch/cuda/_memory_viz.py trace_plot snapshot_layer${FST_LAYER_IDX}.pickle -o snapshot_layer${FST_LAYER_IDX}.html
```
You can now open the resulting HTML file in your browser.

For a healthy run, you should see:

* Brief initial phase with little memory being used. This is the forward pass
  for KV cache checkpointing
* Train of pyramids, one for each cell in the row
* All but the last pyramid has a number of layers proportional to how many
  chunks are in the cell. There are also high but narrow spikes on top of the
  downward slope, one per chunk. The layers correspond to tensors stored in the
  `autograd` graph. If `autograd` saved tensors packing works properly, none of
  them should be as large as the KV cache buffers. The narrow spikes are
  memory required in MHA backward computations.
* The final pyramid corresponds to the cell with the prefill chunk, where
  more memory is required. Its shape depends on internals of SDPA implementations
  in `PyTorch`.
* Memory at the end of the recording should be roughly the same as at the start.
  In particular, GPU memory should not build up across several snapshots


## Implementing New KV Cache Policies

Currently supported KV cache policies are detailed
[here](#kv-cache-policy-and-configuration). We provide a clean and simple
abstraction of KV caching, which makes it simple to implement other policies.
Researchers do not need to care about details such as cache quantization
(just pick a buffer from what we provide) or scaled dot product attention, but
can focus on the essentials. Here, we detail the most important classes to
start from.

### [AttnWeightsKVCache](./keys_values/kvcache/attn_weights.py#L283)

Choose this base class to implement a KV cache policy which makes score-based
decisions. The score values may depend on attention weights (summed over the
query axis). For a concrete example, look at
[H2OKVCache](./keys_values/kvcache/h2o.py#L28). The base class supports a few
features generically:

* Score computations and eviction decision making, by way of
  `next_positions` and `_update`. This is for scores which depend on attention
  weights, so that `update_requires_attn_weights` returns `True`. Overwrite
  `_compute_scores` to implement your score.
* Grace period is implemented generically. All you need to do is to return
  scores for slots outside the grace region in `_compute_scores`.
* Support of replay logging and gradient computation. This should work out of
  the box. If replaying your cache needs more information than stored in
  [AttnWeightsReplayLog](./keys_values/kvcache/attn_weights.py#L39), you need
  to create a subclass.

### [KVCacheWithBuffers](./keys_values/kvcache/basics.py#L42)

Choose this base class to implement a KV cache policy which makes use of one of
the provided buffer strategies (subclasses of
[KVCacheBuffers](./keys_values/kvcache/buffers.py#L128)), and if
`AttnWeightsKVCache` does not work for you or is not needed. The separation
between KV cache policy and buffer strategy is an important aspect of our
abstraction and has a number of direct advantages. Policies and strategies can
be combined at will. Also, we support buffer de-allocation and re-allocation,
which is important to save GPU memory during the backward pass.

### [DefaultKVCache](./keys_values/kvcache/base.py#L383)

Choose this base class to implement a KV cache policy where scaled dot product
attention (SDPA) and position encoding can be factored out, and if
`KVCacheWithBuffers` does not work for you.

Maybe you like to wrap some code which comes with its own buffer strategy.
However, consider extracting the strategy and contribute it separately, so that
other cache policies can make use of it as well.

### [KVCache](./keys_values/kvcache/base.py#L68)

Choose this base class only if all others do not apply. Your code will have to
deal with SDPA, position encoding and buffer strategies. We recommend to use
this base class only to wrap existing monolithic code.
