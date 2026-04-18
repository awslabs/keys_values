# Original Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# Modification Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import sys
import time
import warnings
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from lightning.fabric.plugins import BitsandbytesPrecision

from litgpt.generate.base import sample
from litgpt.prompts import PromptStyle, has_prompt_style, load_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    check_file_size_on_cpu_and_warn,
    check_valid_checkpoint_dir,
    extend_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint,
)

from keys_values.config import Config
from keys_values.data.base import LIT_MODEL_FNAME
from keys_values.kvcache.factory import (
    deallocate_kv_cache_buffers_of_model,
    KVCacheFactory,
)
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT


def next_token(
    gpt_model: GPT,
    x: torch.Tensor,
    **sample_kwargs: Dict[str, Any],
) -> torch.Tensor:
    logits = gpt_model(x)
    _next = sample(logits, **sample_kwargs).to(dtype=torch.int64)
    return _next


def batched_sample(
    logits_stack: torch.Tensor,
    kwargs: Union[dict, list[dict]],
) -> torch.Tensor:
    # Unbind the logits stack into a list of logits.
    logits = [logits_stack] if logits_stack.ndim == 1 else logits_stack.unbind(0)
    logits = [l.unsqueeze(0) for l in logits]
    _kwargs = kwargs if isinstance(kwargs, list) else [kwargs] * len(logits)
    assert len(logits) == len(_kwargs), "logits and kwargs must have the same length."
    return torch.stack(
        [
            sample(l, **sample_args).to(dtype=torch.int64)
            for sample_args, l in zip(_kwargs, logits)
        ],
        dim=0,
    )


def batched_next_token(
    gpt_model: GPT,
    x: torch.Tensor,
    kwargs: Union[dict, list[dict]],
) -> torch.Tensor:
    """

    Args:
        gpt_model: GPT model
        x: Context tokens to be used as input, shape `(batch_size, num)`. When
            used to sample new tokens, we have `num == 1`
        kwargs: Sampling parameters (can be different for each batch dimension)

    Returns:
        New samples corresponding to inputs `x`

    """
    # Run the model on the batch.
    logits_stack = gpt_model(x)

    # Return the next token for each sample in the batch.
    return batched_sample(logits_stack, kwargs=kwargs)


@torch.inference_mode()
def generate_fn(
    model: LongContextInferenceModel,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    stop_tokens: Tuple[List[int], ...] = (),
    include_prompt: bool,
    include_eos: bool,
    deallocate_cache_buffers: bool,
) -> Iterator[torch.Tensor]:
    """
    Generates tokens for a single prompt.

    Args:
        model: The model to use. Must be :class:`LongContextInferenceModel`,
            defining the chunking to be used.
        prompt: The tokenized prompt to generate from.
        max_returned_tokens: The maximum number of new tokens to return. Does not include the prompt tokens.
        temperature: The temp to pass to sample().
        top_k: The top_k to pass to sample().
        top_p: The top_p to pass to sample().
        stop_tokens: A tuple of stop sequences. If any of the sequences are generated, the generation stops early before max_returned_tokens.
        include_prompt: Whether to output the prompt tokens.
        include_eos: Whether to output the stop tokens if generation stops early.
        deallocate_cache_buffers: Whether to deallocate KV cache buffers at
            the end.

    """
    prompt = prompt.flatten()
    prompt_size = prompt.numel()
    if prompt_size == 0:
        raise ValueError("prompt must not be empty")
    sample_kwargs = dict(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    assert (
        max_returned_tokens > prompt_size
    ), f"Not enough space for {prompt_size} prompt tokens in a context length of {max_returned_tokens}."
    model.gpt_model.max_seq_length = max_returned_tokens

    # Yield the prompt if include_prompt is True
    if include_prompt:
        yield prompt

    # Prompt processing. This is dealt with by the long context inference
    # model. Processing is done in chunks, the first one being as long as
    # KV caches permit, subsequent ones (if any) chosen by the model.
    # We need the logits for the final position in order to generate the
    # first token below.
    gpt_model = model.gpt_model
    logits_final_position = model(prompt.unsqueeze(0), targets=None)
    assert logits_final_position.ndim == 3 and logits_final_position.shape[1] == 1, (
        logits_final_position.shape,
    )

    # Generation loop: One token per iteration
    tokens = []
    stop_progress = [0] * len(stop_tokens)
    yielded_idx = 0
    token = None
    for current_idx in range(max_returned_tokens - prompt_size):
        # Generate the token
        if token is None:
            # First token sampled from the final logits output for prompt
            # processing
            token = sample(logits_final_position, **sample_kwargs).to(dtype=torch.int64)
            logits_final_position = None
        else:
            token = next_token(
                gpt_model=gpt_model,
                x=token.view(1, -1),
                **sample_kwargs,
            )
        tokens.append(token)
        int_token = token.item()

        # Check for stop sequences
        # For each stop sequence, we keep a running total of how many are matched in stop_progress.
        # If the current token matches the next token in the stop sequence, we increment the
        # running total and hold off on yielding the token.
        for i, seq in enumerate(stop_tokens):
            if int_token == seq[stop_progress[i]]:
                stop_progress[i] += 1
                if stop_progress[i] == len(seq):
                    if include_eos:
                        yield from tokens[yielded_idx:]
                    return
            else:
                stop_progress[i] = 0

        # Yield tokens that are not part of a stop sequence in progress.
        # If there are no stop sequences, then that's all of them.
        if stop_tokens:
            safe_idx = len(tokens) - max(stop_progress)
        else:
            safe_idx = current_idx + 1  # include the token just generated

        if yielded_idx < safe_idx:
            y_tokens = tokens[yielded_idx:safe_idx]
            yield from y_tokens
            yielded_idx = safe_idx

    # Yield any remaining tokens
    if yielded_idx < len(tokens):
        yield from tokens[yielded_idx:]

    if deallocate_cache_buffers:
        deallocate_kv_cache_buffers_of_model(model.gpt_model)


@torch.inference_mode()
def batched_generate_fn(
    model: LongContextInferenceModel,
    prompts_or_logits: torch.Tensor,
    max_returned_tokens: int,
    *,
    ignore_index: int = -100,
    sample_args: Union[list[dict], dict],
    stop_tokens: Tuple[List[int], ...] = (),
    deallocate_cache_buffers: bool = True,
) -> Iterator[torch.Tensor]:
    """
    Generates tokens for a batch of prompts.

    Prompts may have been processed already, in which case `prompts_or_logits`
    are the logits for the final token position. Even if `prompts_or_logits`
    are prompts, these need not be the start, in that `input_pos` of the KV
    caches may be positive.

    Note: Semantics have changed from `LitGPT` code! For example, the
    prompt is never returned here, and stop sequence tokens are always
    returned.

    Args:
        model: The model to use. Must be :class:`LongContextInferenceModel`,
            defining the chunking to be used.
        prompts_or_logits: Either prompts, shape `(batch_size, prompt_length)`,
            or logits of final token position, shape
            `(batch_size, 1, padded_vocab_size)`. In the latter case, we skip
            prompt processing, assuming the KV caches have been prepared
            appropriately.
        max_returned_tokens: The maximum number of tokens returned
        ignore_index: Token index returned for batch dimensions where generation
            has stopped already
        sample_args: The dictionary of kwargs to pass to sample() for each
            token for each index in the batch.
        stop_tokens: A tuple of stop sequences. If any of the sequences are
            generated, the generation stops early before max_returned_tokens.
            Note that tokens which are part of stop sequences, are still
            returned. They'll be followed by `ignore_index`.
        deallocate_cache_buffers: Whether to deallocate KV cache buffers at
            the end.

    Yields:
        Tensors of shape `(batch_size, 1)`. The entries for batch dimensions
        where generation has stopped, are set to `ignore_index`.

    """
    is_logits = prompts_or_logits.ndim == 3
    gpt_model = model.gpt_model
    input_pos = gpt_model.kv_cache_input_pos()
    if input_pos is None:
        raise AssertionError("model must have KV caches assigned")
    if not is_logits:
        if prompts_or_logits.ndim == 1:
            prompts_or_logits = prompts_or_logits.unsqueeze(0)
        assert (
            prompts_or_logits.ndim == 2
        ), "prompts_or_logits must be a 2D or 3D tensor."
        batch_size, prompt_len = prompts_or_logits.shape
    else:
        if input_pos == 0:
            raise ValueError(
                "Being 3D, prompts_or_logits represents logits, but the KV caches are empty"
            )
        batch_size, prompt_len = prompts_or_logits.shape[0], 0
    if max_returned_tokens <= 0:
        raise ValueError("max_returned_tokens must be > 0")
    if isinstance(sample_args, dict):
        sample_args = [sample_args] * batch_size
    else:
        assert (
            len(sample_args) == batch_size
        ), "sample_args must have the length as the batch size."
    if any(len(x) == 0 for x in stop_tokens):
        raise ValueError(
            "None of the sequences in stop_tokens must be empty:\n" + str(stop_tokens)
        )
    gpt_model.max_seq_length = input_pos + prompt_len + max_returned_tokens

    # Prompt processing (if any)
    if not is_logits:
        # We need the logits for the final chunk in order to generate the
        # first token below. Chunk size does not matter, just must be nonzero.
        logits_final_position = model(prompts_or_logits, targets=None)
    else:
        logits_final_position = prompts_or_logits[:, -1:, :]

    # If `stop_progresses[b][si] > 0`, the past `k = stop_progresses[b][si]`
    # batch dim `b` tokens are equal to the first `k` entries of
    # `stop_tokens[si]`
    stop_progresses = [[0] * len(stop_tokens) for _ in range(batch_size)]
    has_stopped = [False] * batch_size
    stopped_mask = torch.tensor(
        has_stopped,
        dtype=torch.bool,
        device=logits_final_position.device,
    )
    ignore_ind_vec = torch.tensor(
        ignore_index,
        dtype=torch.int64,
        device=logits_final_position.device,
    )

    # Generation loop: One token per iteration
    tokens = None
    for _ in range(max_returned_tokens):
        if logits_final_position is not None:
            tokens = batched_sample(logits_final_position, kwargs=sample_args)
            logits_final_position = None
        else:
            tokens = batched_next_token(
                gpt_model=gpt_model,
                x=tokens,
                kwargs=sample_args,
            )
        int_tokens = [token.item() for token in tokens]

        # Check for stop sequences
        stop_dims = []  # Stop in this iter
        for batch_idx, (int_token, stop_progress) in enumerate(
            zip(int_tokens, stop_progresses)
        ):
            if has_stopped[batch_idx]:
                continue
            for seq_idx, (seq, seq_pos) in enumerate(zip(stop_tokens, stop_progress)):
                assert seq_pos < len(seq)  # Sanity check
                if int_token == seq[seq_pos]:
                    seq_pos += 1
                    if seq_pos >= len(seq):
                        # Full stop sequence matched
                        stop_dims.append(batch_idx)
                        break
                else:
                    # Reset, to 0 or 1
                    seq_pos = int(seq_pos > 0 and int_token == seq[0])
                stop_progress[seq_idx] = seq_pos

        yield torch.where(stopped_mask, ignore_ind_vec, tokens.flatten()).view(-1, 1)
        for batch_idx in stop_dims:
            has_stopped[batch_idx] = True
            stopped_mask[batch_idx] = True
        if all(has_stopped):
            break

    if deallocate_cache_buffers:
        deallocate_kv_cache_buffers_of_model(model.gpt_model)


@torch.inference_mode()
def generate(
    model: LongContextInferenceModel,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id: Optional[int] = None,
    include_prompt: bool = True,
    deallocate_cache_buffers: bool = False,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate
    as many tokens as requested. The implementation of this function is
    modified from A. Karpathy's nanoGPT.

    `model` must be of type :class:`LongContextInferenceModel`, which
    deals with chunking `prompt` into parts if it is too long. For example:

    ```python
    model = LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=None,
        chunk_size=chunk_size,
    )
    ```

    creates the `model` argument from a :class:`GPT` `gpt_model` and
    chunk size `chunk_size`. The first chunk will be as long as KV
    caches support, subsequent chunks will be of length `chunk_size`.
    Here, `gpt_model` must have KV caches assigned.

    Args:
        model: The model to use. Must be :class:`LongContextInferenceModel`,
            defining the chunking to be used.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
            In top-p sampling, the next token is sampled from the highest probability tokens
            whose cumulative probability exceeds the threshold `top_p`. When specified,
            it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
            to sampling the most probable token, while `top_p=1` samples from the whole distribution.
            It can be used in conjunction with `top_k` and `temperature` with the following order
            of application:

            1. `top_k` sampling
            2. `temperature` scaling
            3. `top_p` sampling

            For more details, see https://arxiv.org/abs/1904.09751
            or https://huyenchip.com/2024/01/16/sampling.html#top_p
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
        include_prompt: If true (default) prepends the prompt (after applying the prompt style) to the output.
        deallocate_cache_buffers: Whether to deallocate KV cache buffers at
            the end. Defaults to `False`.

    """
    token_list = list(
        generate_fn(
            include_prompt=include_prompt,
            include_eos=True,
            model=model,
            prompt=prompt,
            max_returned_tokens=max_returned_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_tokens=(([eos_id],) if eos_id is not None else ()),
            deallocate_cache_buffers=deallocate_cache_buffers,
        )
    )

    return torch.cat(token_list) if not len(token_list) == 0 else torch.Tensor()


DEFAULT_CACHE_FACTORY_KWARGS = {
    "name": "lastrec-default",
    "cache_length": 8096,
}


@torch.inference_mode()
def main(
    checkpoint_dir: Path,
    prompt: str = "What food do llamas eat?",
    *,
    cache_factory_kwargs: Optional[Dict[str, Any]] = None,
    sys_prompt: Optional[str] = None,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    prompt_chunksize: int = 16,
    top_k: Optional[int] = 50,
    top_p: float = 1.0,
    temperature: float = 0.8,
    quantize: Optional[
        Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]
    ] = None,
    precision: Optional[str] = None,
    compile: bool = False,
) -> None:
    """Default generation option.

    Generates text samples based on a pre-trained model and tokenizer.

    Args:
        checkpoint_dir: The checkpoint directory to load.
        prompt: The prompt string to use for generating the samples.
        cache_factory_kwargs: Keyword arguments to pass to
            :meth:`KVCacheFactory.create` in order to create KV caches
        sys_prompt: The system prompt to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        prompt_chunksize: If the prompt is longer than the KV cache length,
            prompts are processed in chunks of this size in the prefill phase.
            The larger, the faster the prompt is processed, but a large chunk
            size may lead to suboptimal cache decisions.
        top_k: The number of top most probable tokens to consider in the sampling process.
        top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
            In top-p sampling, the next token is sampled from the highest probability tokens
            whose cumulative probability exceeds the threshold `top_p`. When specified,
            it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
            to sampling the most probable token, while `top_p=1` samples from the whole distribution.
            It can be used in conjunction with `top_k` and `temperature` with the following order
            of application:

            1. `top_k` sampling
            2. `temperature` scaling
            3. `top_p` sampling

            For more details, see https://arxiv.org/abs/1904.09751
            or https://huyenchip.com/2024/01/16/sampling.html#top_p
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to compile the model.
    """
    try:
        from litgpt.constants import _BITANDBYTES_AVAILABLE_NOT_EQUAL_0_42_0
    except ModuleNotFoundError:
        # Earlier version of LitGPT
        from litgpt.utils import _BITANDBYTES_AVAILABLE_NOT_EQUAL_0_42_0

    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())
    if cache_factory_kwargs is None:
        cache_factory_kwargs = DEFAULT_CACHE_FACTORY_KWARGS

    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        if _BITANDBYTES_AVAILABLE_NOT_EQUAL_0_42_0:
            warnings.warn(
                "LitGPT only supports bitsandbytes v0.42.0. This may result in errors when using quantization."
            )
        dtype = {
            "16-true": torch.float16,
            "bf16-true": torch.bfloat16,
            "32-true": torch.float32,
        }[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    checkpoint_path = checkpoint_dir / LIT_MODEL_FNAME
    check_file_size_on_cpu_and_warn(checkpoint_path, fabric.device)

    tokenizer = Tokenizer(checkpoint_dir)
    prompt_style = (
        load_prompt_style(checkpoint_dir)
        if has_prompt_style(checkpoint_dir)
        else PromptStyle.from_config(config)
    )

    prompt = prompt_style.apply(prompt, sys_prompt=sys_prompt)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    fabric.print(
        f"Loading model {str(checkpoint_path)!r} with {config.__dict__}",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        gpt_model = GPT(config)
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            **cache_factory_kwargs,
            gpt_model=gpt_model,
            max_batch_size=1,
        )
    )
    fabric.print(
        f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.",
        file=sys.stderr,
    )
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        gpt_model.max_seq_length = max_returned_tokens
    gpt_model.eval()
    model = LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=None,
        chunk_size=prompt_chunksize,
    )

    if compile:
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        global next_token
        next_token = torch.compile(next_token, mode="reduce-overhead")

    model = fabric.setup_module(model)

    t0 = time.perf_counter()
    load_checkpoint(fabric, gpt_model, checkpoint_path)
    fabric.print(
        f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.",
        file=sys.stderr,
    )

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate(
            model=model,
            prompt=encoded,
            max_returned_tokens=max_returned_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=tokenizer.eos_id,
        )
        t = time.perf_counter() - t0
        fabric.print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        fabric.print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec",
            file=sys.stderr,
        )
    if fabric.device.type == "cuda":
        fabric.print(
            f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB",
            file=sys.stderr,
        )
