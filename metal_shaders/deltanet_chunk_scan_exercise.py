"""
Exercise: chunked DeltaNet's intra-chunk associative scan.

Background
----------
DeltaNet's recurrent state update looks like:

    S_t = a_t * S_{t-1} + b_t

where `a_t` is a decay term (how much of the previous state survives) and
`b_t` is the local update contributed by this timestep's (k_t, v_t, beta_t)
via the delta rule. This is a linear recurrence, and linear recurrences are
associative under the combine operator:

    combine((a_L, b_L), (a_R, b_R)) = (a_R * a_L, a_R * b_L + b_R)

    where L is the earlier (lower-index) pair and R is the later one, i.e.
    "R absorbs L": applying the combined pair to some S_{-1} must equal
    applying L first, then R:
        a_R * (a_L * S_{-1} + b_L) + b_R
      = (a_R * a_L) * S_{-1} + (a_R * b_L + b_R)

Because it's associative, you can scan it in parallel exactly like a sum
scan -- just replace "+" with `combine`. This is the "local associative
update" step chunked DeltaNet does inside each chunk (on the fast, in-simdgroup
path) before chunk-boundary states are combined globally across chunks.

Your job: extend the Hillis-Steele pattern from prefix_sum_exercise.py to
scan PAIRS (a, b) instead of scalars.
"""

import mlx.core as mx
import mlx.core.fast as fast
import numpy as np

N = 32  # one simdgroup's worth of timesteps = one chunk

# ---------------------------------------------------------------------------
# Reference (given): plain sequential recurrence in numpy, S_{-1} = 0.
# ---------------------------------------------------------------------------
def sequential_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    S = np.zeros_like(a)
    prev = 0.0
    for t in range(len(a)):
        prev = a[t] * prev + b[t]
        S[t] = prev
    return S


# ---------------------------------------------------------------------------
# TODO: Metal kernel that computes the same thing with a parallel scan.
#
# Each lane starts holding its own (a_lane, b_lane). At each stride, a lane
# pulls the (a, b) pair from `stride` lanes behind it (its "left" neighbor in
# the scan-so-far) and combines: neighbor is L, this lane's current pair is R.
#
# Remember: after step `stride`, each lane's (a, b) already represents the
# combined effect of a window of `stride` original elements -- so at the next
# larger stride, the neighbor you shuffle in is also already a combined pair,
# not a raw input. That's what makes it O(log n) instead of O(n).
# ---------------------------------------------------------------------------
source = """
uint lane = thread_index_in_simdgroup;
T a = decay[lane];
T b = update[lane];

for (uint stride = 1; stride < 32; stride *= 2) {
    // TODO 1: shuffle in the neighbor pair (a_L, b_L) from `stride` lanes back
    T a_left = simd_shuffle_up(a, stride);
    T b_left = simd_shuffle_up(b, stride);

    if (lane >= stride) {
        // TODO 2: combine (a_left, b_left) as L with (a, b) as R:
        //   combine((a_L,b_L), (a_R,b_R)) = (a_R*a_L, a_R*b_L + b_R)
        // Careful: `b` depends on the OLD `a`, so compute b before
        // overwriting a, or stash the old value.

        T new_a =  a * a_left;
        T new_b = a * b_left + b;
        a = new_a;
        b = new_b;
    }
}

out_state[lane] = b;   // S_lane assuming S_{-1} = 0
out_decay[lane] = a;   // cumulative decay from chunk start through lane
                        // (needed later to fold in the real S_{-1} from the
                        // previous chunk: S_lane_true = a * S_prev_chunk + b)
"""

kernel = fast.metal_kernel(
    name="deltanet_chunk_scan",
    input_names=["decay", "update"],
    output_names=["out_state", "out_decay"],
    source=source,
)


def chunk_scan(a: mx.array, b: mx.array):
    outs = kernel(
        inputs=[a, b],
        template=[("T", mx.float32)],
        output_shapes=[a.shape, a.shape],
        output_dtypes=[a.dtype, a.dtype],
        grid=(N, 1, 1),
        threadgroup=(N, 1, 1),
    )
    return outs[0], outs[1]


if __name__ == "__main__":
    mx.random.seed(0)
    # decay terms in (0, 1) -- e.g. sigmoid(forget gate); updates ~ small deltas
    a = mx.sigmoid(mx.random.normal((N,)))
    b = 0.1 * mx.random.normal((N,))

    ref = sequential_reference(np.array(a), np.array(b))
    print("sequential reference S_t:", np.round(ref, 4))

    state, cum_decay = chunk_scan(a, b)
    mx.eval(state, cum_decay)
    print("parallel scan       S_t:", np.round(np.array(state), 4))

    if np.allclose(np.array(state), ref, atol=1e-4):
        print("MATCH: your associative scan is correct.")
    else:
        print("MISMATCH: fill in the TODOs in `source`.")

    # Sanity check on the side output: cumulative decay from chunk start
    # should equal the running product of `a`.
    expected_decay = np.cumprod(np.array(a))
    print("cumulative decay  a_t:", np.round(np.array(cum_decay), 4))
    print("expected (cumprod)  :", np.round(expected_decay, 4))
