import logging
import math 
import os
from turtle import shape

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

MAX_TILE_SIZE = 256
MIN_TILE_SIZE = 32

logger = logging.getLogger(__name__)

# BLOCK_Q, BLOCK_K, num_warps, num_stages
_h100_default_config = {
    (torch.float32, 64): (128, 32, 4, 3),
    (torch.float32, 128): (32, 64, 4, 3),
    (torch.float32, 256): (32, 32, 4, 3),
    (torch.bfloat16, 64): (128, 128, 4, 3),
    (torch.bfloat16, 128): (128, 64, 8, 3),
    (torch.bfloat16, 256): (64, 32, 4, 3),
    (torch.float16, 64): (128, 128, 4, 3),
    (torch.float16, 128): (128, 128, 8, 3),
    (torch.float16, 256): (64, 32, 4, 3),
}

_a100_default_config = {
    (torch.float32, 64): (128, 32, 4, 3),
    (torch.float32, 128): (128, 32, 4, 3),
    (torch.float32, 256): (64, 16, 4, 3),
    (torch.bfloat16, 64): (128, 64, 4, 3),
    (torch.bfloat16, 128): (128, 64, 8, 3),
    (torch.bfloat16, 256): (32, 64, 4, 3),
    (torch.float16, 64): (128, 64, 4, 3),
    (torch.float16, 128): (128, 64, 8, 3),
    (torch.float16, 256): (32, 64, 4, 3),
}

def _get_default_config_fwd(head_dim, dtype) -> tuple[int, int, int, int]:
    default_config = None

    if head_dim <= 256 and torch.cuda.get_device_capability() >= (9, 0):  # H100
        if dtype == torch.float32:
            default_config = (64, 64, 4, 3)
        else:
            default_config = (128, 64, 4, 3)
        default_config = _h100_default_config.get((dtype, head_dim), default_config)
    elif head_dim <= 256 and torch.cuda.get_device_capability() >= (8, 0):  # A100
        if dtype == torch.float32:
            default_config = (64, 64, 4, 3)
        else:
            default_config = (128, 64, 4, 3)
        default_config = _a100_default_config.get((dtype, head_dim), default_config)
    else:  # modest hardware or extremely large head_dim
        if dtype == torch.float32:
            default_config = (32, 16, 4, 3)
        else:
            default_config = (64, 32, 4, 3)

    return default_config

def strides(t):
    assert t is not None
    return [t.stride(i) for i in range(t.ndim)]

def fwd_configs_pruner(configs, nargs, HEAD_DIM, DTYPE, **kwargs):
    min_size, max_size = 16, 256
    min_pipeline, max_pipeline = 1, 3
    min_warps, max_warps = 1, 8

    if HEAD_DIM == 64:
        min_pipeline = 2
    elif HEAD_DIM == 128:
        max_size = 128
        min_size = 32
        max_pipeline = 3
        max_warps = 4
    elif HEAD_DIM == 256:
        max_size = 128
        min_size = 32
        max_pipeline = 2
        max_warps = 4

    configs = [i for i in configs if min_size <= i.kwargs["TILE_K_SIZE"] <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs["TILE_Q_SIZE"] <= max_size]
    configs = [
        i for i in configs if min_pipeline <= i.kwargs["PIPELINING"] <= max_pipeline
    ]
    configs = [i for i in configs if min_warps <= i.num_warps <= max_warps]

    default_config = _get_default_config_fwd(HEAD_DIM, DTYPE)
    if default_config is not None:
        configs += [
            triton.Config(
                dict(
                    PIPELINING=default_config[3],
                    TILE_Q_SIZE=default_config[0],
                    TILE_K_SIZE=default_config[1],
                    V_PRELOAD=V_PRELOAD,
                ),
                num_warps=default_config[2],
                num_stages=default_config[3],
            )
            for V_PRELOAD in (True, False)
        ]

    logger.warning(f"Start benchmarking forward streaming_attention {len(configs) = }")
    return configs

# fmt: off
@triton.heuristics(
    dict(
        RCP_LN2=lambda _: math.log2(math.e),
        V_PRELOAD=lambda _: True,
    )
)
@triton.jit
def _self_attn_fwd(
    Q: tl.tensor, Kt: tl.tensor, V: tl.tensor, L: tl.tensor, #
    O: tl.tensor,  #
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,  #
    stride_kb: int, stride_kh: int, stride_kk: int, stride_kt: int,  #
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,  #
    stride_ob: int, stride_oh: int, stride_ot: int, stride_ok: int, #
    lens_stride: int,
    T: int,  #
    PRESCALE: tl.constexpr,  #
    TIME_BUCKET:  int,  #
    LEN_PRESENT: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    SM_SCALE: tl.constexpr,  #
    DTYPE:  tl.constexpr,  #
    TILE_Q_SIZE: tl.constexpr,  #
    TILE_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
    V_PRELOAD: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    q_tile_idx = tl.program_id(2)
    q_token_idx = q_tile_idx * TILE_Q_SIZE

    if LEN_PRESENT:
        seq_len = tl.load(L + batch * lens_stride)
        seq_len = min(seq_len, T) 
        need_q_mask = q_token_idx + TILE_Q_SIZE
    else: 
        seq_len = T
        need_q_mask = False

    if seq_len <= q_token_idx:
        return
    
    qbatch_head_offset = batch * stride_qb + head * stride_qh
    q_tile_ptr = tl.make_block_ptr(
        base=Q + qbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_qt, stride_qk),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1,0),
    )

    kbatch_head_offset = batch * stride_kb + head * stride_kh
    kt_tile_ptr =tl.make_block_ptr(
        base=Kt + kbatch_head_offset,
        shape=(HEAD_DIM, T),
        strides=(stride_kk, stride_kt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_K_SIZE),
        order=(0,1),
    )

    # What is T
    vbatch_head_offest = batch * stride_vb + head * stride_vh
    v_tile_ptr = tl.make_block_ptr(
        base = V + vbatch_head_offest,
        shape = (T, HEAD_DIM),
        strides = (stride_vt, stride_vk),
        offsets=(0,0),
        block_shape=(TILE_K_SIZE, HEAD_DIM),
        order=(1,0),
    )

    # Why minus dtype minus infinity? and what is mi li, acc?
    # acc is multiplied with alpha and .where
    # m_i might be the minimum
    m_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32)
    acc = tl.zeros([TILE_Q_SIZE, HEAD_DIM], dtype=tl.float32)

    q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)

    q_tile = tl.load(
        q_tile_ptr,
        boundary_check=(0,),
    )

    softmax_scale: tl.constexpr = tl.cast(SM_SCALE * RCP_LN2, q_tile.dtype)
    tile_k_arange = tl.arange(0, TILE_K_SIZE)

    if PRESCALE:
        q_tile *= softmax_scale

    # What is cdiv;maxtile = integer of size?
    max_tile = tl.cdiv(seq_len, TILE_K_SIZE)
    # Iterate over each elem in max_tile
    for kv_tile_idx in tl.range(
        0, max_tile, num_stages=PIPELINING
    ):
        last_iter = kv_tile_idx == max_tile - 1 
        # kv_token_idx
        kv_token_idx = kv_tile_idx * TILE_K_SIZE

        if last_iter: 
            kt_tile = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
                boundary_check=(1,)
            )
        else:
            kt_tile = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
            )
        if V_PRELOAD:
            if last_iter:
                v_tile = tl.load(
                    tl.advance(v_tile_ptr, (kv_token_idx, 0))
                    boundary_check=(0,),
                )
            else: 
                v_tile = tl.load(
                    tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                )

        # partial qk multipliacation I guess
        qk = tl.dot(
            q_tile, kt_tile, input_precision=INPUT_PRECISION, out_dtype=tl.float32
        )

        # So i guess online softmax is assumed, what is PRESCALE though
        if not PRESCALE:
            qk *= softmax_scale

        # adds mask to qks if last_iter
        if last_iter:
            # So we add a range to a the index makes sense
            kv_indices = kv_token_idx + tile_k_arange

            mask = (
                kv_indices[None, :] < seq_len
            )

            qk = tl.where(mask, qk, tl.cast(-float("inf"), qk.dtype))

        # Now we get into the online softmax stuf forreal
        # Dont really understand what m_ij is with max(imum) logic unclear
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])

        # Aligns with my basic understanding of online softm
        l_ij = tl.sum(p,1)
        alpha = tl.math.exp2(qk, m_ij[:, None])

        if not V_PRELOAD:
            if last_iter:
                v_tile = tl.load(
                    tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                    boundary_check=(0,),
                )
            else: 
                v_tile = tl.load(
                    tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                )
            
        # Some type of accumulated form with online softmax included now  
        acc = tl.dot(
            p.to(v_tile.dtype),
            v_tile, 
            acc, 
            input_precision=INPUT_PRECISION,
            out_dtype=tl.float32
        )
        m_i = m_ij

    # Final accumulated version
    acc = acc / l_i[:, None]
    # Real query mask instead of negative inf for softmax afaik
    if need_q_mask:
        q_lens_mask = (
            q_tile_indices[:, None] < seq_len
        )
        acc.where(q_lens_mask, acc, 0.0)

    # Lets go! the lines I get
    obatch_head_offset = batch * stride_ob + head * stride_oh
    o_tile_ptr = tl.make_block_ptr(
        base=O+obatch_head_offset,
        shape=(T,HEAD_DIM),
        strides=(stride_ot, stride_ok),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1,0),
    )

    # And Done
    tl.store(
        o_tile_ptr,
        acc.to(o_tile_ptr.type.element_ty),
        boundary_check=(0,),
    )

# Resumen: A lot of offsets, gotta look into preload, prescale,
# online softmax logic and *t, *k strides because I dont understand all dimensions

# What does kwargs["L"] stand for? what was resetonly planned for?
def autotune_prehook(kwargs, reset_only=False):
    if kwargs["L"] is not None:
        kwargs["L"].add_(kwargs["q"].size(2))


def autotune_posthook(kwargs, exception=None):
    if kwargs["L"] is not None:
        kwargs["L"].add_(-kwargs["q"].size(2))

# We seem to add and subtract time from "L" ¿L?

streaming_forward = triton.heuristics(
    dict(
        PIPELININ=lambda _:1,
        TILE_Q_SIZE=lambda _:54,
        TILE_K_SIZE=lambda _: 64,
    )
)(_self_attn_fwd)


# Rest Copy and pasted from https://github.com/alexdremov/kernels/blob/main/src/self_attention/kernel.py
streaming_forward_autotune = triton.autotune(
    configs=[
        triton.Config(
            dict(
                PIPELINING=pipe,
                TILE_Q_SIZE=tile_q,
                TILE_K_SIZE=tile_k,
                V_PRELOAD=V_PRELOAD,
            ),
            num_warps=num_warps,
            num_stages=pipe,
        )
        for num_warps in [4, 8]
        for pipe in [1, 2]
        for tile_q in [
            2**i
            for i in range(
                int(math.log2(MIN_TILE_SIZE) + 0.1),
                int(math.log2(MAX_TILE_SIZE) + 0.1) + 1,
            )
        ]
        for tile_k in [
            2**i
            for i in range(
                int(math.log2(MIN_TILE_SIZE) + 0.1),
                int(math.log2(MAX_TILE_SIZE) + 0.1) + 1,
            )
        ]
        for V_PRELOAD in (True, False)
    ],
    key=["HEAD_DIM", "INPUT_PRECISION", "TIME_BUCKET", "DTYPE"],
    prune_configs_by=dict(early_config_prune=fwd_configs_pruner),
    pre_hook=autotune_prehook,
    post_hook=autotune_posthook,
)(_self_attn_fwd)

# Not bothering to write benchmarking logic