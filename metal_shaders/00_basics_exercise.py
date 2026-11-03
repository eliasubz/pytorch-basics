"""
Exercise 0: the absolute basics of writing a Metal kernel body.

What even is this code?
------------------------
Metal Shading Language (MSL) is basically C++ with a few extra keywords.
When you call mx.fast.metal_kernel(), MLX runs ONE COPY of your `source`
string per "thread", all at the same time, on the GPU. This is the SPMD
model (Single Program, Multiple Data): the code is identical for every
thread, but each thread gets a different index and so touches different
data.

Think of it like this Python, but every iteration of the loop actually runs
truly in parallel, on its own tiny core:

    for i in range(N):
        out[i] = q[i] + k[i]     # <-- this line is your kernel `source`

A few C-isms you need before anything else:
  - every statement ends with `;`
  - variables need a type before their name: `float x = 1.0;`  not `x = 1.0`
  - `T` here is a template placeholder for whatever dtype you pass in via
    `template=[("T", mx.float32)]` -- MLX substitutes it for you.
  - inputs/outputs arrive as pointers: `q` is not "an array" like in Python,
    it's a pointer to the start of a block of GPU memory. `q[i]` means
    "the float sitting at position i after that pointer". Indexing works
    exactly like Python/C list indexing otherwise.
  - MLX auto-generates the function signature (the `kernel void foo(...)`
    part with buffer bindings) from `input_names`/`output_names`. You only
    write the BODY -- the part that would go between `{` and `}`.

How do I know which index I am?
--------------------------------
Every thread can read built-in variables that tell it "who am I":
  - thread_position_in_grid.x -- this thread's global index, 0..N-1.
    This is the one you want 95% of the time for simple elementwise kernels.
  - thread_index_in_simdgroup -- index (0..31) within a 32-wide simdgroup.
    Only matters once you start doing simdgroup-level tricks like shuffles
    (that's what prefix_sum_exercise.py and deltanet_chunk_scan_exercise.py
    are about -- come back to those after this file clicks).

Work through step 0 (given, just run it), then fill in steps 1 and 2.
"""

import mlx.core as mx
import mlx.core.fast as fast
import numpy as np

N = 8  # small on purpose so you can eyeball every value


def run_kernel(source: str, input_names, output_names, inputs):
    kernel = fast.metal_kernel(
        name="basics_kernel",
        input_names=input_names,
        output_names=output_names,
        source=source,
    )
    out = kernel(
        inputs=inputs,
        template=[("T", mx.float32)],
        output_shapes=[inputs[0].shape],
        output_dtypes=[inputs[0].dtype],
        grid=(N, 1, 1),
        threadgroup=(N, 1, 1),
    )[0]
    mx.eval(out)
    return out


# ---------------------------------------------------------------------------
# Step 0 (given): copy input to output, untouched. This is the smallest
# possible kernel -- just to see how a thread reads "its" element and writes
# "its" element.
# ---------------------------------------------------------------------------
copy_source = """
uint i = thread_position_in_grid.x;
out[i] = q[i];
"""

if __name__ == "__main__":
    q = mx.arange(N).astype(mx.float32)
    out = run_kernel(copy_source, ["q"], ["out"], [q])
    print("step 0 (copy):     q =", np.array(q), " out =", np.array(out))
    assert mx.array_equal(out, q).item(), "copy kernel should be a no-op"
    print("  OK\n")

    # -----------------------------------------------------------------
    # Step 1 (TODO): elementwise add. out[i] = q[i] + k[i]
    #
    # Fill in `add_source` below. Model it exactly on copy_source above,
    # just with two inputs instead of one.
    # -----------------------------------------------------------------
    add_source = """
    uint i = thread_position_in_grid.x;
    out[i] = 0;  // TODO: replace with q[i] + k[i]
    """

    q = mx.arange(N).astype(mx.float32)
    k = mx.full((N,), 10.0)
    out = run_kernel(add_source, ["q", "k"], ["out"], [q, k])
    expected = q + k
    print("step 1 (add):      expected =", np.array(expected), " got =", np.array(out))
    if mx.allclose(out, expected).item():
        print("  OK\n")
    else:
        print("  MISMATCH: fill in add_source\n")

    # -----------------------------------------------------------------
    # Step 2 (TODO): scale-and-shift. out[i] = q[i] * 2 + 1
    #
    # This is a preview of the "decay * state + update" shape you'll need
    # for the DeltaNet exercise -- but with a plain constant instead of a
    # per-thread decay term, and no dependency between threads yet.
    # -----------------------------------------------------------------
    scale_shift_source = """
    uint i = thread_position_in_grid.x;
    out[i] = 0;  // TODO: replace with q[i] * 2 + 1
    """

    q = mx.arange(N).astype(mx.float32)
    out = run_kernel(scale_shift_source, ["q"], ["out"], [q])
    expected = q * 2 + 1
    print("step 2 (scale+1):  expected =", np.array(expected), " got =", np.array(out))
    if mx.allclose(out, expected).item():
        print("  OK\n")
    else:
        print("  MISMATCH: fill in scale_shift_source\n")

    print("Once all three say OK, move on to prefix_sum_exercise.py --")
    print("that's the same `out[i] = ...` idea, but now each thread also")
    print("reads its NEIGHBORS' values via simd_shuffle, instead of only")
    print("ever touching its own index i.")
