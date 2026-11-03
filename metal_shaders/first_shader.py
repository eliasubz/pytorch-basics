import mlx.core as mx
import mlx.core.fast as fast



source = """
uint elem = thread_position_in_grid.x;
out[elem] = q[elem] * k[elem];
"""

kernel = fast.metal_kernel(
    name="delta_net_tile",
    input_names=["q", "k"],
    output_names=["out"],
    source=source,
)

# Call directly with MLX arrays
q = mx.random.normal((128,))
k = mx.random.normal((128,))
out = kernel(
    inputs=[q, k],
    template=[("T", mx.float32)],
    output_shapes=[(128,)],
    output_dtypes=[mx.float32],
    grid=(128, 1, 1),
    threadgroup=(32, 1, 1),
)[0]
mx.eval(out)