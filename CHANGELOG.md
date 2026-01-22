# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/).

<a name="v0.1.0"></a>
## [0.1.0] - 2026-??-??

### New Features
* Key-value cache abstraction to extend `LitGPT` models for long-context inference
  and fine-tuning
* Supports sparse attention and selective key-value cache policies
* Key-value cache policies `dense`, `lastrec`, `h2o`, `qh2o`, `h2o-vlen`, `qh2o-vlen`
* Fine-tuning of models on long context data with KV caches embedded
* Quantization of KV cache buffers and activation checkpoints to 4 or 8 bits
* Memory-efficient `MultiHeadSelfAttention` with explicit `backward`
* `RoPE` position encoding with `YaRN` scaling

### Documentation Updates
* Documentation of concepts in [README.md](./README.md)
