import time

import mlx.core as mx
import mlx.core.fast as fast
import numpy as np

# Puzzle 1 
def add_spec(x):
    "This is the spec that you should implement. Uses typing to define sizes."
    return x + 10.

add_kernel_source = """
uint lane = thread_index_in_simdgroup;
z[lane] = x[lane] + 10;
"""
_add_kernel = fast.metal_kernel(
    name="add_kernel",
    input_names=["x"],
    output_names=["z"],
    source=add_kernel_source,
)

N = 32

def add_kernel(x: mx.array):
    outs = _add_kernel(
        inputs=[x],
        template=[("T", mx.float32)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(N, 1, 1),
        threadgroup=(N, 1, 1),
    )
    return outs[0]


# Puzzle 2 
def outer_vec_spec(x, y):
    return x[None, :] + y[:, None]

# not clear we have thread_position_in_grid, thread_position_in_thread_group and thread_index_in_simd
outer_vec_source = """
uint i = thread_position_in_grid.y;
uint j = thread_position_in_grid.x;

z[i * num_cols + j] = x[j] + y[i];
"""

_outer_vec_kernel = fast.metal_kernel(
    name="outer_vec_kernel",
    input_names=["x", "y", "num_cols"],
    output_names=["z"],
    source=outer_vec_source,
)

def outer_vec_kernel(x: mx.array, y: mx.array):
    outs = _outer_vec_kernel(
        inputs=[x, y, x.shape[0]],
        template=[("T", mx.float32)],
        output_shapes=[(y.shape[0] ,x.shape[0])],
        output_dtypes=[x.dtype],
        grid=(x.shape[0], y.shape[0], 1),
        threadgroup=(32, 4, 1),
    )
    return outs[0]
# puzzle 3 
# Now we will make this more parallel and work for even bigger inputs



def check(kernel_fn, spec_fn, *inputs, atol=1e-4, warmup=True):
    """Run kernel_fn and spec_fn on the same inputs, compare, and time both."""
    if warmup:
        # first call pays for Metal source compilation, not just the kernel
        # itself -- run once and throw it away so the timing below is fair.
        mx.eval(kernel_fn(*inputs))

    t0 = time.perf_counter()
    got = kernel_fn(*inputs)
    mx.eval(got)
    t1 = time.perf_counter()
    expected = spec_fn(*inputs)
    mx.eval(expected)
    t2 = time.perf_counter()

    ok = mx.allclose(got, expected, atol=atol).item()
    status = "PASS" if ok else "FAIL"
    print(f"{status}: {kernel_fn.__name__}  "
          f"(kernel {1000*(t1-t0):.3f} ms, spec {1000*(t2-t1):.3f} ms)")
    if not ok:
        print("  got     :", np.array(got))
        print("  expected:", np.array(expected))
    return ok


if __name__ == "__main__":
    x = mx.arange(N).astype(mx.float32)
    check(add_kernel, add_spec, x)
    x = mx.arange(10000).astype(mx.float32)
    y = mx.arange(128).astype(mx.float32)
    check(outer_vec_kernel, outer_vec_spec, x, y)



# cheat sheet
"""
Attribute,Meaning,Value in Threadgroup 0,Value in Threadgroup 1
threadgroup_position_in_grid,Block ID (blockIdx.x),0,1
thread_position_in_threadgroup,Local Thread ID (threadIdx.x),65,65
thread_index_in_simdgroup,Lane ID (laneId / % 32),1,1
simdgroup_index_in_threadgroup,Which SIMD-group (threadIdx.x / 32),2 (the 3rd SIMD-group),2 (the 3rd SIMD-group)
thread_position_in_grid,Global Thread ID,65,193
"""

