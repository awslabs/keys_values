# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/).

<a name="v0.1.0"></a>
## [0.1.0] - 2026-02-20

Thanks to all contributors:
@mseeger, @Jantory, @wesk

### New Features
* Key-value cache abstraction to extend `LitGPT` models for long-context inference
  and fine-tuning
* Supports sparse attention and selective key-value cache policies (such as H2O)
* Key-value cache policies `dense`, `lastrec`, `h2o`, `qh2o`, `h2o-vlen`, `qh2o-vlen`
* Fine-tuning of models on long context data with KV caches embedded
* Fast scaled dot product attention via `flex_attention`
* Quantization of KV cache buffers and activation checkpoints to 4 or 8 bits
* Memory-efficient `MultiHeadSelfAttention` with explicit `backward`
* `RoPE` position encoding with `YaRN` scaling
* Fine-tuning scripts supporting `LoRA` and CPU offloading, with distributed
  data parallelism across multiple GPUs

### Documentation Updates
* Documentation of concepts in [README.md](./README.md)

<a name="v0.1.0"></a>
## [0.2.0] - 2026-06-01

Thanks to all contributors:
@mseeger, @vihangp, @Jantory

### New Features
* Refactor evaluation script so it can run with baseline checkpoints ([#123](https://github.com/awslabs/keys_values/pull/123))
* Changes for CPU--GPU transfer, leading to substantial speed-up ([#120](https://github.com/awslabs/keys_values/pull/120))
* Allocate inference replay cache buffers once and reuse them for each shard ([#118](https://github.com/awslabs/keys_values/pull/118))
* Load generation config from checkpoint, and improve CL args ([#116](https://github.com/awslabs/keys_values/pull/116))
* Evaluation scripts can also write generated samples to files ([#114](https://github.com/awslabs/keys_values/pull/114))
* More tests for FlashInfer SDPA ([#113](https://github.com/awslabs/keys_values/pull/113))
* Unit tests for new fused operators ([#110](https://github.com/awslabs/keys_values/pull/110))
* Refactoring and cleaning up fused operators ([#109](https://github.com/awslabs/keys_values/pull/109))
* Clean up data part for training state, make it generic. Fixes a bug as well ([#108](https://github.com/awslabs/keys_values/pull/108))
* Fused kernels and removal of torch cuda synchronize ([#105](https://github.com/awslabs/keys_values/pull/105))
* New evaluation script which iterates over several setups ([#106](https://github.com/awslabs/keys_values/pull/106))
* Store training state and resume training from stored state. Fix bug in `SFTDataset.__getitem__` ([#103](https://github.com/awslabs/keys_values/pull/103))
* Implement own DDP used without CPU offloading ([#100](https://github.com/awslabs/keys_values/pull/100))
* Refactoring (new attention module). Additional tests for FlashInfer SDPA kernel ([#95](https://github.com/awslabs/keys_values/pull/95))
* Support gradient clipping. Simplify loss normalization (sum losses over chunks, normalize at end) ([#94](https://github.com/awslabs/keys_values/pull/94))
* Add FlashInfer CUDA kernels and Triton score-sum for efficient attention weight computation
* Simplify `batched_generate_fn` and `SampleBasedMetricsEvaluator` ([#90](https://github.com/awslabs/keys_values/pull/90))
* Code for supporting sample-based metrics for evaluation ([#81](https://github.com/awslabs/keys_values/pull/81))
* Baseline SDPA returning attention weights, calling FlexAttention 2x ([#78](https://github.com/awslabs/keys_values/pull/78))
* Refactoring `Helmet`: Structure of metadata dictionary ([#73](https://github.com/awslabs/keys_values/pull/73))
* Simplify `LayerInputCheckpoints` by providing cell ranges at construction. Each cell range gets its own buffers ([#70](https://github.com/awslabs/keys_values/pull/70))
* Make sure that 1D token_positions special case is exploited. Alternative to sorting if 3D. FlexAttention does not work with sliding_window_size ([#69](https://github.com/awslabs/keys_values/pull/69))
* KV cache checkpoint objects are now reused across several `GradientAccumulator.run` calls. And fixed CL args for memory pinning ([#67](https://github.com/awslabs/keys_values/pull/67))
* Support CPU memory pinning for activation and KV cache checkpointing ([#63](https://github.com/awslabs/keys_values/pull/63))
* Allow for intermediate checkpoints ([#57](https://github.com/awslabs/keys_values/pull/57))
* Mechanism to reduce number of graphs compiled for FlexAttention. Eliminate dependency of graphs on scale ([#58](https://github.com/awslabs/keys_values/pull/58))
* Several LoRA variants (DoRA; RMS normalization). Also fix load/save LoRA checkpoints ([#56](https://github.com/awslabs/keys_values/pull/56))
* Extended and reorganized tooling: Store selected weights/grads. Tracking of tensors for NaNs and large values ([#51](https://github.com/awslabs/keys_values/pull/51))
* CPU offloading of (quantized) KV cache buffers ([#47](https://github.com/awslabs/keys_values/pull/47))
* Modernize `longcontext_eval` script (no more Fabric). New evaluation data loader. Test data iterator for this case. Debug code for eval script ([#39](https://github.com/awslabs/keys_values/pull/39))

### Bug Fixes
* Fix bug concerning smart-lastrec ([#102](https://github.com/awslabs/keys_values/pull/102))
* Small fixes and additions ([#74](https://github.com/awslabs/keys_values/pull/74))
* Fix issue of changing `dtype` for score buffers in `AttnWeightsKVCache`, due to `fabric.setup` ([#38](https://github.com/awslabs/keys_values/pull/38))
