"""
Exercise: parallel prefix sum (scan) within a single simdgroup.

Background
----------
A simdgroup on Apple GPUs is 32 threads ("lanes") that execute in lockstep and
can exchange register values directly via simd_shuffle_* intrinsics, without
touching threadgroup (shared) memory. That makes it the cheapest place to do
a scan.

Two ways to compute it:
  1. simd_prefix_exclusive_sum(val)   -- built-in, hardware-accelerated.
  2. Manual Hillis-Steele scan using simd_shuffle_up -- what you'll implement.
     Understanding (2) is what lets you later swap "+" for a custom combine
     operator (see deltanet_chunk_scan_exercise.py).

Run this file as-is first (part A works out of the box) to see the
reference behavior, then fill in the TODOs in part B and re-run to check
yourself against part A.
"""

import mlx.core as mx
import mlx.core.fast as fast
import numpy as np

N = 32  # exactly one simdgroup

# ---------------------------------------------------------------------------
# Part A (given): built-in simd_prefix_exclusive_sum, as ground truth.
# ---------------------------------------------------------------------------
builtin_source = """
uint lane = thread_index_in_simdgroup;
T val = inp[lane];
out[lane] = simd_prefix_exclusive_sum(val);
"""

builtin_kernel = fast.metal_kernel(
    name="prefix_sum_builtin",
    input_names=["inp"],
    output_names=["out"],
    source=builtin_source,
)


def prefix_sum_builtin(a: mx.array) -> mx.array:
    return builtin_kernel(
        inputs=[a],
        template=[("T", mx.float32)],
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        grid=(N, 1, 1),
        threadgroup=(N, 1, 1),
    )[0]


# ---------------------------------------------------------------------------
# Part B (TODO): manual Hillis-Steele inclusive scan via simd_shuffle_up.
#
# Idea: at each step `stride` (1, 2, 4, 8, 16), every lane >= stride adds in
# the value currently held by the lane `stride` behind it. After log2(32)=5
# steps every lane holds the INCLUSIVE sum of all lanes up to and including
# itself.
#
# simd_shuffle_up(value, delta) returns the `value` held by lane
# (this_lane - delta). For lanes where this_lane - delta < 0, the result is
# unspecified/garbage -- you must guard with `if (lane >= stride)`.
# ---------------------------------------------------------------------------
manual_source = """
uint lane = thread_index_in_simdgroup;
T val = inp[lane];

for (uint stride = 1; stride < 32; stride *= 2) {
    // TODO 1: fetch the value held by the lane `stride` behind this one
    T neighbor = /* TODO: simd_shuffle_up(???, ???) */ 0;

    // TODO 2: only lanes with an actual neighbor at this stride should add
    if (/* TODO: condition */ false) {
        val = val + neighbor;
    }
}

// `val` is now the INCLUSIVE scan. The builtin above is EXCLUSIVE, so
// subtract the original element to compare against it directly.
out[lane] = val - inp[lane];
"""

manual_kernel = fast.metal_kernel(
    name="prefix_sum_manual",
    input_names=["inp"],
    output_names=["out"],
    source=manual_source,
)


def prefix_sum_manual(a: mx.array) -> mx.array:
    return manual_kernel(
        inputs=[a],
        template=[("T", mx.float32)],
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        grid=(N, 1, 1),
        threadgroup=(N, 1, 1),
    )[0]


if __name__ == "__main__":
    mx.random.seed(0)
    a = mx.random.normal((N,))

    ref = prefix_sum_builtin(a)
    mx.eval(ref)
    print("builtin exclusive scan:", np.round(np.array(ref), 3))

    out = prefix_sum_manual(a)
    mx.eval(out)
    print("manual  exclusive scan:", np.round(np.array(out), 3))

    if mx.allclose(ref, out, atol=1e-4).item():
        print("MATCH: your manual scan is correct.")
    else:
        print("MISMATCH: fill in the TODOs in `manual_source`.")
