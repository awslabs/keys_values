# KeysAndValues: Efficient Language Model Inference, Fine-tuning, and Key-value Caching

This library provides implementations of advanced key-value caching for
efficient long context inference and fine-tuning with large language models.
It sits on top of [LitGPT](https://github.com/Lightning-AI/litgpt/tree/main).


## Setup for Development

For the moment, this library requires minor modifications to `LitGPT` contained
in a fork. You can obtain this code as follows:

```bash
git clone git@github.com:mseeger/litgpt.git
cd litgpt
git checkout valkeyrie_new
```

Next, you need to create a virtual environment:

```bash
python3 -m venv keyval_venv
. keyval_venv/bin/activate
pip install --upgrade pip
cd ${LITGPT_PATH}
pip install -e .[all,test,extra]
cd ${KEYS_VALUES_PATH}
pip install -e .
```

Here, replace `${LITGPT_PATH}` with the source path of the `LitGPT` fork and
`${KEYS_VALUES_PATH}` with the source path of `keys_values`.

Run the tests in order to check whether the installation worked:

```bash
cd ${KEYS_VALUES_PATH}
pytest test/
```


## Example: Fine-tuning on LongBench V2

This should work on a reasonably sized GPU instance, e.g. P4:

```bash
cd ${KEYS_VALUES_PATH}
python3 keys_values/__main__.py finetune_long_lora Qwen/Qwen2.5-0.5B --data LongBenchV2 --data.max_seq_length 100000 --data.metadata_dir /home/ubuntu/out/finetune/longcontext_lora/data --out_dir /home/ubuntu/out/finetune/longcontext_lora --precision bf16-true --kv_cache.cache_length 16384 --kv_cache.chunk_size 1024 --kv_cache.verbose some --kv_cache.layers_per_cell 1 --train.save_interval 10 --train.micro_batch_size 4 --train.global_batch_size 4 --eval.interval 10
```


## Run GPU Memory Profiling

This is based on https://pytorch.org/blog/understanding-gpu-memory-1/. It shows
how to profile GPU memory usage during the backward pass.

GPU memory profiling is activated with the CL option `--record_gpu_memory_snapshots 50000`.
The number is the `max_entries` argument for `torch.cuda.memory._record_memory_history`.
Pickle files are written to:

* `${OUT_DIR}/gpu_memory_snapshots/iteration${ITER}/snapshot_initial.pickle`:
  From start of iteration until backward over top-most layer. Includes the
  forward pass for layer input checkpoints and KV cache logs, as well as the
  backward for the head model.
* `${OUT_DIR}/gpu_memory_snapshots/iteration${ITER}/snapshot_layer${FST_LAYER_IDX}.pickle`:
  Backward over one row of cells. Here, `FST_LAYER_IDX` the index of the first
  layer for the row of cells.

Here, `OUT_DIR` is given by the CL option `--out_dir ${OUT_DIR}`, `ITER` is the
iteration number. Copy the snapshot files from the GPU instance.

To watch a snapshot, you have two ways. Try to upload the pickle file to their web
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


## Relevant Parameters for Long Context Fine-Tuning

**Note**: We plan to have accurate formulae for estimating GPU memory
requirements given parameters. These can then be used to optimize the
configuration before running the code. This work has not been finished
yet.

Here, we give some general advice about parameters on which GPU memory usage
and computation speed depends. Each gradient computation on a batch is split
into two stages:

* Forward pass in inference mode: This is storing layer input checkpoints
  to CPU and logs all KV cache decisions. At the end of this stage, all KV
  cache buffers are deallocated (but not the model weights).
* Backward pass to compute gradients: This is an outer loop over rows of
  cells (top down). For each row, we first run another forward pass in
  inference mode to compute KV cache checkpoints at cell boundaries, storing
  them on CPU. We don't store these in the first stage forward pass, because
  this would require too much CPU memory. Then, we iterate over cells from
  right to left. For each cell, we run `autograd` with additional memory-saving
  tricks to compute gradients.

Apart from model size, the most important parameter dictating GPU memory
requirements is the KV cache size (`--kv_cache.cache_length <...>`). The
larger the cache size, the closer exact multi-head self attention is
approximated, and the faster gradient computation runs. The required GPU
memory requirements in both stages scale linearly with KV cache size.

**Note**: Our code supports different KV cache lengths for each layer,
but this is not yet enabled for the CLI.

Another important argument is `--kv_cache.name`, selecting both cache
policy and quantization. A value has the form `<cname>-<qname>`, where
`<cname>` determines the cache policy, `<qname>` the quantization. Here
are values for `<qname>`:

* "default": No quantization. KV cache buffers stored in same datatype as
  weights (set by `--precision`).
* "torch-quantized8", "bnb-quantized8", "ao-quantized8": KV cache buffers
  quantized to 8 bits. For 16 bit weights, this saves a factor of 2.
* "bnb-quantized4", "ao-quantized4": KV cache buffers quantized to 4 bits.
  For 16 bit weights, this saves a factor of 4. At this moment, neither
  `bitsandbytes` nor `torchao` support 4 bit quantization on CPU.

Next, here are further relevant parameters determining GPU memory usage in
the second stage:

* `--kv_cache.chunk_size`: Length of chunks in which the sequence batch is
  split for processing (except the first chunk, whose length is close to the
  cache length). Must be substantially smaller than cache length. A larger
  chunk size means fewer chunks, therefore faster processing. However, for
  larger chunks, the KV cache policy is used less often, and so the
  approximation to exact MHA can be worse. GPU memory requirements also grow
  with chunk size.
* `--kv_cache.layers_per_cell`: Second stage GPU memory requirements depend
  linearly on this number. It states how many layers are processed in a cell.
  Larger values mean less sequential processing, so faster computation.
  Ignoring anything else, choose this as large as GPU memory permits.
  Note that the CPU memory for layer input checkpoints scales inverse
  linearly with this number.
