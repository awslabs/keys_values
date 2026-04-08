"""Compare attention weights: double FlexAttention vs FlashInfer+Triton vs PyTorch reference.

This test creates identical Q, K, V inputs and computes attention weight sums
using three methods:
1. PyTorch reference (matmul + softmax + sum) — ground truth
2. FlashInfer SDPA + Triton score-sum kernel (our method)
3. Double FlexAttention (baseline PR #78 method)

Usage:
  /fsx/pvihang/flash_infer_integration/keys_values/venvs/v2_venv/bin/python test_compare_weights.py
"""

import torch
import math


def pytorch_reference_weights(Q, K, scale, input_pos, token_positions, n_kv_heads, group_size):
    """Ground truth: full matmul + softmax + causal mask + sum.

    Args:
        Q: [batch, q_len, n_head, head_dim]
        K: [batch, kv_len, n_kv_heads, head_dim]
        scale: softmax scale factor
        input_pos: starting query position
        token_positions: [batch, n_kv_heads, kv_len] int positions
        n_kv_heads, group_size: GQA parameters

    Returns:
        W: [batch, n_kv_heads, kv_len] float32 attention weight sums
    """
    batch, q_len, n_head, hd = Q.shape
    kv_len = K.shape[1]

    # Expand to [batch, n_head, q_len, hd] and [batch, n_head, kv_len, hd]
    Q_4d = Q.permute(0, 2, 1, 3).float()  # [B, H, Q, D]
    K_exp = K.permute(0, 2, 1, 3).float()  # [B, Hkv, KV, D]
    K_exp = K_exp.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
    K_exp = K_exp.reshape(batch, n_head, kv_len, hd)

    # Scores: [batch, n_head, q_len, kv_len]
    scores = torch.matmul(Q_4d, K_exp.transpose(-1, -2)) * scale

    # Causal mask
    q_positions = torch.arange(q_len, device=Q.device) + input_pos  # [q_len]
    # token_positions: [batch, n_kv_heads, kv_len] -> expand to [batch, n_head, kv_len]
    tp_exp = token_positions.unsqueeze(2).expand(-1, -1, group_size, -1)
    tp_exp = tp_exp.reshape(batch, n_head, kv_len)

    # mask[b, h, q, k] = True if kv_pos <= query_pos
    mask = tp_exp[:, :, None, :] <= q_positions[None, None, :, None]
    scores.masked_fill_(~mask, float('-inf'))

    # Softmax + sum
    weights = torch.softmax(scores, dim=-1)  # [B, H, Q, KV]
    # Sum over queries, then group heads -> [B, Hkv, KV]
    weights = weights.reshape(batch, n_kv_heads, group_size, q_len, kv_len)
    W = weights.sum(dim=(2, 3))
    return W.float()


def triton_weights(Q, K, scale, input_pos, token_positions, n_kv_heads, group_size):
    """Our method: FlashInfer SDPA for LSE + Triton score-sum kernel."""
    from keys_values.flashinfer_wrapper import triton_score_sum

    batch, q_len, n_head, hd = Q.shape
    kv_len = K.shape[1]

    # Compute LSE via PyTorch (same as what FlashInfer sdpa_prefill returns)
    Q_4d = Q.permute(0, 2, 1, 3).float()
    K_exp = K.permute(0, 2, 1, 3).float()
    K_exp = K_exp.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
    K_exp = K_exp.reshape(batch, n_head, kv_len, hd)
    scores = torch.matmul(Q_4d, K_exp.transpose(-1, -2)) * scale

    # Apply causal mask for LSE computation too
    q_positions = torch.arange(q_len, device=Q.device) + input_pos
    tp_exp = token_positions.unsqueeze(2).expand(-1, -1, group_size, -1)
    tp_exp = tp_exp.reshape(batch, n_head, kv_len)
    mask = tp_exp[:, :, None, :] <= q_positions[None, None, :, None]
    scores.masked_fill_(~mask, float('-inf'))

    lse = torch.logsumexp(scores, dim=-1)  # [B, H, Q] in ln scale
    lse_log2 = (lse / math.log(2)).permute(0, 2, 1).float()  # [B, Q, H] in log2 scale

    W = triton_score_sum(
        Q, K, lse_log2, scale, n_kv_heads, group_size,
        token_positions=token_positions,
        input_pos=input_pos,
    )
    return W


def flexatt_weights(Q, K, V, scale, input_pos, token_positions, n_kv_heads, group_size):
    """Baseline: double FlexAttention from PR #78."""
    from keys_values.flex_attention import (
        sdpa_flexatt_with_attn_weights,
        FlexAttentionArgs,
    )

    batch, q_len, n_head, hd = Q.shape
    kv_len = K.shape[1]

    # FlexAttention expects [batch, n_head, q_len, hd] for Q
    # and [batch, n_kv_heads, kv_len, hd] for K, V
    Q_4d = Q.permute(0, 2, 1, 3).contiguous()
    K_4d = K.permute(0, 2, 1, 3).contiguous()
    V_4d = V.permute(0, 2, 1, 3).contiguous()

    flexatt_args = FlexAttentionArgs(
        forward_return_lse=True,
        extend_kv=False,
    )

    _, attn_weights = sdpa_flexatt_with_attn_weights(
        flexatt_args=flexatt_args,
        query=Q_4d,
        key=K_4d,
        value=V_4d,
        scale_factor=scale,
        attention_logit_softcapping=None,
        input_pos=input_pos,
        token_positions=token_positions,
    )

    return attn_weights.float()


def run_comparison():
    torch.manual_seed(42)
    device = 'cuda'

    # Qwen3-4B config
    batch = 1
    q_len = 16
    n_head = 32
    n_kv_heads = 8
    hd = 128
    group_size = n_head // n_kv_heads
    kv_len = 64
    input_pos = 48  # simulates non-prefill: query starts at pos 48
    scale = 1.0 / math.sqrt(hd)

    # Contiguous token positions: 0..kv_len-1
    token_positions = torch.arange(kv_len, device=device).unsqueeze(0).unsqueeze(0)
    token_positions = token_positions.expand(batch, n_kv_heads, -1).contiguous().int()

    Q = torch.randn(batch, q_len, n_head, hd, device=device, dtype=torch.bfloat16)
    K = torch.randn(batch, kv_len, n_kv_heads, hd, device=device, dtype=torch.bfloat16)
    V = torch.randn(batch, kv_len, n_kv_heads, hd, device=device, dtype=torch.bfloat16)

    print(f"Config: batch={batch}, q_len={q_len}, kv_len={kv_len}, n_head={n_head}, n_kv={n_kv_heads}, hd={hd}")
    print(f"input_pos={input_pos}, scale={scale:.6f}")
    print(f"token_positions: contiguous [0..{kv_len-1}]")
    print()

    # 1. PyTorch reference
    with torch.no_grad():
        W_ref = pytorch_reference_weights(Q, K, scale, input_pos, token_positions, n_kv_heads, group_size)

    # 2. Triton score-sum
    with torch.no_grad():
        W_triton = triton_weights(Q, K, scale, input_pos, token_positions, n_kv_heads, group_size)

    # 3. Double FlexAttention
    try:
        with torch.no_grad():
            W_flex = flexatt_weights(Q, K, V, scale, input_pos, token_positions, n_kv_heads, group_size)
        has_flex = True
    except Exception as e:
        print(f"FlexAttention failed: {e}")
        has_flex = False

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Compare Triton vs reference
    diff_triton = (W_triton - W_ref).abs()
    print(f"\nTriton vs PyTorch reference:")
    print(f"  Max abs diff:  {diff_triton.max().item():.6f}")
    print(f"  Mean abs diff: {diff_triton.mean().item():.6f}")
    rel_diff = diff_triton / (W_ref.abs() + 1e-8)
    print(f"  Max rel diff:  {rel_diff.max().item():.6f}")
    print(f"  Mean rel diff: {rel_diff.mean().item():.6f}")

    if has_flex:
        # Compare FlexAttention vs reference
        diff_flex = (W_flex - W_ref).abs()
        print(f"\nFlexAttention vs PyTorch reference:")
        print(f"  Max abs diff:  {diff_flex.max().item():.6f}")
        print(f"  Mean abs diff: {diff_flex.mean().item():.6f}")
        rel_diff_flex = diff_flex / (W_ref.abs() + 1e-8)
        print(f"  Max rel diff:  {rel_diff_flex.max().item():.6f}")
        print(f"  Mean rel diff: {rel_diff_flex.mean().item():.6f}")

        # Compare Triton vs FlexAttention
        diff_tf = (W_triton - W_flex).abs()
        print(f"\nTriton vs FlexAttention:")
        print(f"  Max abs diff:  {diff_tf.max().item():.6f}")
        print(f"  Mean abs diff: {diff_tf.mean().item():.6f}")

        # Show actual weight samples
        print(f"\nSample weights (kv_head=0, first 10 KV positions):")
        print(f"  Reference:      {W_ref[0, 0, :10].tolist()}")
        print(f"  Triton:         {W_triton[0, 0, :10].tolist()}")
        print(f"  FlexAttention:  {W_flex[0, 0, :10].tolist()}")

        # Check eviction decisions: which top-K entries differ?
        keep_ratio = 0.5
        keep_k = int(kv_len * keep_ratio)
        topk_ref = W_ref[0, 0].topk(keep_k).indices.sort().values
        topk_triton = W_triton[0, 0].topk(keep_k).indices.sort().values
        topk_flex = W_flex[0, 0].topk(keep_k).indices.sort().values

        triton_match = (topk_triton == topk_ref).all().item()
        flex_match = (topk_flex == topk_ref).all().item()
        triton_flex_match = (topk_triton == topk_flex).all().item()

        print(f"\nEviction decisions (keep top {keep_k} of {kv_len}):")
        print(f"  Triton matches reference:      {triton_match}")
        print(f"  FlexAttention matches reference: {flex_match}")
        print(f"  Triton matches FlexAttention:   {triton_flex_match}")

        if not flex_match:
            ref_set = set(topk_ref.tolist())
            flex_set = set(topk_flex.tolist())
            only_ref = ref_set - flex_set
            only_flex = flex_set - ref_set
            print(f"  Positions kept by reference but not FlexAtt: {sorted(only_ref)}")
            print(f"  Positions kept by FlexAtt but not reference: {sorted(only_flex)}")

    # Non-contiguous positions (simulating H2O eviction)
    print("\n" + "=" * 70)
    print("TEST 2: Non-contiguous positions (simulating H2O eviction)")
    print("=" * 70)

    # Simulate: some positions evicted, gaps in token_positions
    kept_positions = torch.tensor([0, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 47, 48, 49, 50,
                                    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
                                   device=device, dtype=torch.int32)
    kv_len2 = len(kept_positions)
    token_positions2 = kept_positions.unsqueeze(0).unsqueeze(0).expand(batch, n_kv_heads, -1).contiguous()

    K2 = torch.randn(batch, kv_len2, n_kv_heads, hd, device=device, dtype=torch.bfloat16)
    V2 = torch.randn(batch, kv_len2, n_kv_heads, hd, device=device, dtype=torch.bfloat16)

    print(f"kv_len={kv_len2}, non-contiguous positions: {kept_positions[:10].tolist()}...{kept_positions[-5:].tolist()}")
    print()

    with torch.no_grad():
        W_ref2 = pytorch_reference_weights(Q, K2, scale, input_pos, token_positions2, n_kv_heads, group_size)
        W_triton2 = triton_weights(Q, K2, scale, input_pos, token_positions2, n_kv_heads, group_size)

    diff2 = (W_triton2 - W_ref2).abs()
    print(f"Triton vs PyTorch reference (non-contiguous):")
    print(f"  Max abs diff:  {diff2.max().item():.6f}")
    print(f"  Mean abs diff: {diff2.mean().item():.6f}")

    try:
        with torch.no_grad():
            W_flex2 = flexatt_weights(Q, K2, V2, scale, input_pos, token_positions2, n_kv_heads, group_size)

        diff_flex2 = (W_flex2 - W_ref2).abs()
        print(f"\nFlexAttention vs PyTorch reference (non-contiguous):")
        print(f"  Max abs diff:  {diff_flex2.max().item():.6f}")
        print(f"  Mean abs diff: {diff_flex2.mean().item():.6f}")

        # Eviction decisions
        keep_k2 = int(kv_len2 * 0.5)
        topk_ref2 = W_ref2[0, 0].topk(keep_k2).indices.sort().values
        topk_triton2 = W_triton2[0, 0].topk(keep_k2).indices.sort().values
        topk_flex2 = W_flex2[0, 0].topk(keep_k2).indices.sort().values

        print(f"\nEviction decisions (keep top {keep_k2} of {kv_len2}):")
        print(f"  Triton matches reference:      {(topk_triton2 == topk_ref2).all().item()}")
        print(f"  FlexAttention matches reference: {(topk_flex2 == topk_ref2).all().item()}")
    except Exception as e:
        print(f"\nFlexAttention failed on non-contiguous: {e}")


if __name__ == "__main__":
    run_comparison()
