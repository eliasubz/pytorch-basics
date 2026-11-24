"""
Day 04 — EXERCISE (target: ~45 min): Tensor parallelism (Megatron-style MLP) from scratch.
Run:  python logrun.py day04_tp.py

Data parallel (Days 1-3): every rank has DIFFERENT data, the SAME (full or sharded) weights.
Tensor parallel (today):  every rank has the SAME data, a DIFFERENT slice of each weight matrix.

# ---------------------------------------------------------------------------
# LOG: fill in as soon as something breaks (logrun.py records the raw error for you)
# Symptom:
# Cause:
# Fix:
# Rule:
# Category: memory | comms | process | math | testing
# ---------------------------------------------------------------------------
"""
import os, socket
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Your ring collectives (working 1-D versions from Days 1-3)
# ---------------------------------------------------------------------------
def _ring(chunks, rank, W, first_send, reduce):
    right, left = (rank + 1) % W, (rank - 1) % W
    for step in range(W - 1):
        s, r = (first_send - step) % W, (first_send - step - 1) % W
        buf = torch.empty_like(chunks[r])
        req = dist.isend(chunks[s].contiguous(), right); dist.recv(buf, left); req.wait()
        chunks[r].add_(buf) if reduce else chunks[r].copy_(buf)


def ring_all_reduce(t):
    """In place, any shape (t must be contiguous)."""
    rank, W = dist.get_rank(), dist.get_world_size()
    chunks = t.view(-1).tensor_split(W)
    _ring(chunks, rank, W, rank, reduce=True)        # afterwards rank owns chunk rank+1
    _ring(chunks, rank, W, rank + 1, reduce=False)
    return t


def ring_reduce_scatter(flat):
    """1-D in, returns my summed chunk (chunk `rank`). Input untouched."""
    assert flat.dim() == 1
    rank, W = dist.get_rank(), dist.get_world_size()
    chunks = flat.clone().tensor_split(W)
    _ring(chunks, rank, W, rank - 1, reduce=True)
    return chunks[rank]


def ring_all_gather(my_chunk, out):
    """1-D: writes every rank's chunk into `out` (chunk r at position r)."""
    assert out.dim() == 1
    rank, W = dist.get_rank(), dist.get_world_size()
    chunks = out.tensor_split(W)
    chunks[rank].copy_(my_chunk.reshape(-1))
    _ring(chunks, rank, W, rank, reduce=False)
    return out


# ---------------------------------------------------------------------------
# Part 1: N-D collectives along any dimension
# ---------------------------------------------------------------------------
def all_gather_dim(x, dim):
    """All-gather along ANY dim: result shape = x.shape with x.shape[dim] * W.
    Idea: move `dim` to the front (movedim + contiguous) so each rank's piece is ONE contiguous
    block; run the 1-D ring_all_gather on flat views; move the dim back.
    Check yourself: why does tensor_split along dim=1 NOT give contiguous chunks?
    """
    # Whats the Difference swapdim and movedim
    W = dist.get_world_size()
    x0 = torch.movedim(x, dim, 0).contiguous()
    out = torch.empty((x0.shape[0] * W, *x0.shape[1:]))
    ring_all_gather(x0.view(-1), out.view(-1))
    return out.movedim(0, dim)

# learned input is a shard of x and allgather works inplace although it also outputs out.


def reduce_scatter_dim(x, dim):
    """Reduce-scatter along ANY dim: returns my chunk (number `rank`) of the sum,
    i.e. sum_over_ranks(x).tensor_split(W, dim)[rank]. Same movedim trick.
    You may assume x.shape[dim] is divisible by W (assert it).
    """
    r, W = dist.get_rank(), dist.get_world_size()
    x0 = x.movedim(dim, 0).contiguous()
    assert x0.shape[0] % W == 0
    out = ring_reduce_scatter(x0.reshape(-1))
    out = out.reshape(x0.tensor_split(W)[r].shape)
    return out.movedim(0, dim)
    


# ---------------------------------------------------------------------------
# Part 2: the two parallel linear layers (forward only)
# ---------------------------------------------------------------------------
def column_parallel_forward(x, W_cols, gather_output):
    """W is split along its OUTPUT dim (columns): rank r holds W[:, r-th block].
    x is the FULL input, the same on every rank.
    - What does x @ W_cols give you? Is any communication needed?
    - gather_output=True: return the full (N, out) result (all_gather_dim along the last dim).
    """
    # W: [D, r-th block]
    # x: [:, D]
    out = x @ W_cols       # [:, r*s : (r+1)*s]
    # I would say no comuncation is needed.
    return all_gather_dim(out, -1) if gather_output else out
    


def row_parallel_forward(x_cols, W_rows):
    """W is split along its INPUT dim (rows): rank r holds W[r-th block, :].
    x_cols is the input split along its LAST dim the same way (rank r has columns block r).
    - Write the full x @ W as a sum over blocks: x @ W = sum_r x_cols_r @ W_rows_r
    - Each rank computes one term. Which collective turns the terms into the full result?
    """
    # W: [r*s : (r+1)*s, :]
    # x: [:, r*s : (r+1)*s]
    # again no comunication is necessary
    out = x_cols @ W_rows
    return ring_all_reduce(out.contiguous())

# wasnt clear on trying exactly this. I was always changing the dimesnions a bit because I did not trust my first intuition on the outer matmul


# ---------------------------------------------------------------------------
# Part 3: one TP training step for  loss = mean((relu(x @ W1) @ W2 - y)^2)
#   W1: column-parallel (each rank owns H/W hidden units)
#   W2: row-parallel    (each rank owns the matching H/W rows)
# ---------------------------------------------------------------------------
def tp_mlp_step(x, y, W1_cols, W2_rows, lr):
    """Forward + manual backward + SGD on your shards. Returns (loss as float, dx).
    Every rank has the SAME full x and y.

    Forward:  h_pre = column-parallel(x, W1) WITHOUT gathering -> relu -> row-parallel(h, W2)
    Loss:     mean over N*O. Is it already identical on every rank, or do you need a collective?
    Backward (manual, like Day 1), and for each line ask: is the result complete or partial?
        d_out, dW2 (shape (H/W, O)), d_h, d_hpre, dW1 (shape (D, H/W)),
        dx = d_hpre @ W1_cols.T  -> complete or partial?   (Test 3 checks dx against autograd)
    Update:   W1_cols -= lr * dW1 ; W2_rows -= lr * dW2   (in place!)

    Questions:
     - How many collectives per step in total? Compare with DP (Day 1). What gets communicated:
       weights/gradients or activations? What does that mean for which network TP needs?
     - Why do weight gradients need NO communication here, unlike in DP?
     - Megatron calls the all-reduce in forward "g" and the one in backward "f". Where exactly
       are they in your code?
    """
    r, W = dist.get_rank(), dist.get_world_size()
    # rememeber mean is sum, 
    h_pre = column_parallel_forward(x, W1_cols, False)    # [N, s]
    h = torch.clamp_min(h_pre, 0)                   
    out = row_parallel_forward(h, W2_rows)       # [N, O]
    diff = out - y
    loss = (diff ** 2).sum() / diff.numel()

    # W2: [s, O]
    # h:  [N, s]
    dout = 2 * diff / diff.numel() # dL / d_out, bc d_dif = dout, bc dy = 0
    ldW2 = h.T @ dout # dL / dW1

    dh_pre = (dout @ W2_rows.T) * (0 < h_pre)  # [N, H] 
    ldW1 = x.T @ dh_pre        # dL/dW1
    dx = dh_pre @ W1_cols.T

    W1_cols -= lr * ldW1
    W2_rows -= lr * ldW2

    dx = ring_all_reduce(dx)
    return loss, dx


# I did it!!!!
# Mistakes:
# - div by x.numel() but its diff.numel()
# - tried to reduce sharded weights but they are fine how they are no comms are necessary
# - forgot about ring_all_reduce for dx. Differenet coms are still not 100% intuitive
# - accidently used ldW1/2 leaf nodes for back prop dh and dx but that does not make sense. Weight differentials had no effect on partial derivatives of activations.
# tried to reduce 










# ---------------------------------------------------------------------------
# Harness (don't edit)
# ---------------------------------------------------------------------------
D, H, O, N = 6, 12, 3, 32


def make_problem():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, D, generator=g)
    y = torch.randn(N, O, generator=g)
    W1 = torch.randn(D, H, generator=g) * 0.3
    W2 = torch.randn(H, O, generator=g) * 0.3
    return x, y, W1, W2


def reference_step(x, y, W1, W2, lr):
    x = x.clone().requires_grad_(); W1 = W1.clone().requires_grad_(); W2 = W2.clone().requires_grad_()
    loss = ((torch.relu(x @ W1) @ W2 - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        return loss.item(), x.grad, W1 - lr * W1.grad, W2 - lr * W2.grad


def worker(rank, port, steps, lr):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, timeout=timedelta(seconds=30))
    torch.set_default_dtype(torch.float64)
    W = WORLD
    gen = torch.Generator().manual_seed(100 + rank)

    # Test 1: N-D all-gather along every dim
    a = torch.randn(4, 3, 2, generator=gen)
    for dim in range(a.dim()):
        parts = [torch.empty_like(a) for _ in range(W)]; dist.all_gather(parts, a)
        print("ag", all_gather_dim(a, dim))
        print("dist", torch.cat(parts, dim))
        assert torch.equal(all_gather_dim(a, dim), torch.cat(parts, dim)), f"all_gather_dim wrong for dim={dim}"
    # ... and reduce-scatter along every dim
    b = torch.randn(4, 8, 12, generator=gen)
    total = b.clone(); dist.all_reduce(total)
    for dim in range(b.dim()):
        assert torch.allclose(reduce_scatter_dim(b, dim), total.tensor_split(W, dim)[rank]), \
            f"reduce_scatter_dim wrong for dim={dim}"
    if rank == 0: print("Test 1 passed: N-D all-gather + reduce-scatter")

    # Test 2: column- and row-parallel forward == full matmul
    x, y, W1, W2 = make_problem()
    W1_cols = W1.tensor_split(W, dim=1)[rank].clone()
    W2_rows = W2.tensor_split(W, dim=0)[rank].clone()
    assert torch.allclose(column_parallel_forward(x, W1_cols, gather_output=True), x @ W1), "column-parallel wrong"
    h = torch.relu(x @ W1)
    h_cols = h.tensor_split(W, dim=1)[rank]
    assert torch.allclose(row_parallel_forward(h_cols, W2_rows), h @ W2), "row-parallel wrong"
    if rank == 0: print("Test 2 passed: column- and row-parallel linear")

    # Test 3: TP training == single-process training (loss, input grad, weights)
    rW1, rW2 = W1.clone(), W2.clone()
    for step in range(steps):
        loss, dx = tp_mlp_step(x, y, W1_cols, W2_rows, lr)
        ref_loss, ref_dx, rW1, rW2 = reference_step(x, y, rW1, rW2, lr)
        assert abs(loss - ref_loss) < 1e-9, f"step {step}: loss mismatch"
        assert torch.allclose(dx, ref_dx), f"step {step}: dx mismatch (input gradient)"
    assert torch.allclose(W1_cols, rW1.tensor_split(W, 1)[rank]), "W1 shard diverged"
    assert torch.allclose(W2_rows, rW2.tensor_split(W, 0)[rank]), "W2 shard diverged"
    if rank == 0:
        print("Test 3 passed: TP training == single-process training")
        print(f"All tests passed. Last loss {loss:.4f}")
    dist.destroy_process_group()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    mp.spawn(worker, args=(free_port(), 10, 0.1), nprocs=WORLD)
