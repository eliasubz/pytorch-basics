
import torch
import triton
import triton.language as tl

@triton.jit
def gated_delta_net_kernel(
    q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, o_ptr,
    stride_qb, stride_qh, stride_qt, stride_qk,
    stride_kb, stride_kh, stride_kt, stride_kk,
    stride_vb, stride_vh, stride_vt, stride_vv,
    stride_ab, stride_ah, stride_at,
    stride_bb, stride_bh, stride_bt,
    stride_ob, stride_oh, stride_ot, stride_ov,
    T, K: tl.constexpr, V: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    k_offsets = tl.arange(0, BLOCK_K)
    v_offsets = tl.arange(0, BLOCK_V)

    k_mask = k_offsets < K
    v_mask = v_offsets < V

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    q_base = q_ptr + pid_b * stride