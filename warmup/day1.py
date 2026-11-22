"""
Day 01 — SOLUTION: ring all-reduce from scratch + data-parallel SGD with hand-written backprop.
Run:  python day01_solution.py
"""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

WORLD = 4
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Part 1: ring all-reduce  (reduce-scatter phase + all-gather phase)
# ---------------------------------------------------------------------------
def ring_all_reduce(t: torch.Tensor) -> torch.Tensor:
    rank, W = dist.get_rank(), dist.get_world_size()
    right, left = (rank + 1) % W, (rank - 1) % W
    flat = t.reshape(-1)  # view: in-place writes land in t
    chunks = list(flat.tensor_split(W))
    buf = [torch.empty_like(c) for c in chunks]

    # Reduce-scatter: after W-1 steps, rank r holds the full sum of chunk (r+1) % W
    for step in range(W - 1):
        send_idx = (rank - step) % W
        recv_idx = (rank - step - 1) % W
        req = dist.isend(chunks[send_idx].contiguous(), right)
        dist.recv(buf[recv_idx], left)
        req.wait()
        chunks[recv_idx] += buf[recv_idx]

    # All-gather: circulate the finished chunks
    for step in range(W - 1):
        send_idx = (rank - step + 1) % W
        recv_idx = (rank - step) % W
        req = dist.isend(chunks[send_idx].contiguous(), right)
        dist.recv(buf[recv_idx], left)
        req.wait()
        chunks[recv_idx].copy_(buf[recv_idx])
    return t


# ---------------------------------------------------------------------------
# Part 2: manual forward/backward for  loss = mean((relu(x @ W1) @ W2 - y)^2)
# ---------------------------------------------------------------------------
def forward_backward(x, y, W1, W2, n_global):
    """Returns (loss_contribution, dW1, dW2). Loss is normalised by the GLOBAL batch
    size so that summing over ranks gives exactly the full-batch gradient."""
    h_pre = x @ W1                    # [n, H]
    h = h_pre.clamp_min(0)            # relu
    out = h @ W2                      # [n, O]
    diff = out - y
    loss = (diff ** 2).sum() / (n_global * y.shape[1])

    d_out = 2 * diff / (n_global * y.shape[1])   # dL/dout
    dW2 = h.T @ d_out                             # [H, O]
    d_h = d_out @ W2.T                            # [n, H]
    d_hpre = d_h * (h_pre > 0)                    # relu'
    dW1 = x.T @ d_hpre                            # [D, H]
    return loss, dW1, dW2


# ---------------------------------------------------------------------------
# Part 3: data-parallel training step
# ---------------------------------------------------------------------------
def dp_step(x_shard, y_shard, W1, W2, n_global, lr):
    loss, dW1, dW2 = forward_backward(x_shard, y_shard, W1, W2, n_global)
    ring_all_reduce(dW1)               # grads already globally normalised -> plain sum
    ring_all_reduce(dW2)
    loss_t = torch.tensor([loss.item()])
    ring_all_reduce(loss_t)
    W1 -= lr * dW1
    W2 -= lr * dW2
    return loss_t.item()


# ---------------------------------------------------------------------------
# Harness (don't edit)
# ---------------------------------------------------------------------------
def make_problem(N=64, D=8, H=16, O=3):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, D, generator=g)
    y = torch.randn(N, O, generator=g)
    W1 = torch.randn(D, H, generator=g) * 0.3
    W2 = torch.randn(H, O, generator=g) * 0.3
    return x, y, W1, W2


def reference(steps, lr):
    x, y, W1, W2 = make_problem()
    W1.requires_grad_(); W2.requires_grad_()
    losses = []
    for _ in range(steps):
        loss = ((torch.relu(x @ W1) @ W2 - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            W1 -= lr * W1.grad; W2 -= lr * W2.grad
            W1.grad = None; W2.grad = None
        losses.append(loss.item())
    return losses, W1.detach(), W2.detach()


def worker(rank, steps, lr):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29511")
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    torch.set_default_dtype(torch.float64)

    # Test 1: ring all-reduce matches dist.all_reduce (odd size -> uneven chunks)
    t = torch.randn(1001, generator=torch.Generator().manual_seed(rank))
    expected = t.clone(); dist.all_reduce(expected)
    ring_all_reduce(t)
    assert torch.allclose(t, expected), f"rank {rank}: ring all-reduce mismatch"

    # Test 2: manual backprop matches autograd
    x, y, W1, W2 = make_problem()
    _, dW1, dW2 = forward_backward(x, y, W1, W2, x.shape[0])
    a1, a2 = W1.clone().requires_grad_(), W2.clone().requires_grad_()
    ((torch.relu(x @ a1) @ a2 - y) ** 2).mean().backward()
    assert torch.allclose(dW1, a1.grad) and torch.allclose(dW2, a2.grad), "backprop mismatch"

    # Test 3: DP training == single-process full-batch training
    shard = x.shape[0] // WORLD
    xs, ys = x[rank * shard:(rank + 1) * shard], y[rank * shard:(rank + 1) * shard]
    losses = [dp_step(xs, ys, W1, W2, x.shape[0], lr) for _ in range(steps)]
    ref_losses, rW1, rW2 = reference(steps, lr)
    assert torch.allclose(W1, rW1) and torch.allclose(W2, rW2), "DP weights diverged"
    assert all(abs(a - b) < 1e-9 for a, b in zip(losses, ref_losses)), "DP loss mismatch"
    if rank == 0:
        print(f"All tests passed. loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(20, 0.1), nprocs=WORLD)