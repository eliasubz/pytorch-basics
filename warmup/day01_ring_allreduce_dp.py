"""
Day 01 — EXERCISE (target: 30-40 min): ring all-reduce from scratch + data-parallel SGD with hand-written backprop.
Run:  python day01_solution.py
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

WORLD = 4
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Part 1: ring all-reduce  (reduce-scatter phase + all-gather phase)
# ---------------------------------------------------------------------------
def ring_all_reduce(t: torch.Tensor) -> torch.Tensor:
    """In-place SUM all-reduce using only point-to-point ops (dist.isend / dist.recv).
    1. Split the flat tensor into W chunks (tensor_split handles uneven sizes).
    2. Reduce-scatter: W-1 steps. Each step: send one chunk to the right neighbour,
       receive one from the left, add it into your copy. Which index do you send at step s?
    3. All-gather: W-1 more steps circulating the fully-reduced chunks (copy, don't add).
    Hint: use isend + recv + wait, a blocking send on every rank at once can deadlock.
    Bonus question: how many bytes does each rank send in total? Compare to a naive
    "everyone sends everything to rank 0" scheme.
    """
    # t: (N,)
    r, W = dist.get_rank(), dist.get_world_size()
    # Split the tensor into W chunks
    # I want the writes to land in t
    flat = t.view(-1)
    t_chunks = list(torch.tensor_split(flat, W)) # splits t in the last dimension

    # t_chunks: (W, N//W) or its different tensors 

    # Reduce-scatter 
    right = (r + 1) % W
    left = (r - 1) % W
    # I want each rank own its own rth chunk. This means they send r-1%W first and iterate backwards
    src_chunk_idx = (r-1) % W      # Mabye I would like to not change the dst_chunk in order for us to make sure we send the chunk we just recieved
    # print(r, "chunks before", t_chunks,"and sends to", right, "and gets from", left)
    for i in range(W-1):
        # send chunk
        dst_chunk_idx = src_chunk_idx
        sending_chunk = t_chunks[dst_chunk_idx]
        req = dist.isend(sending_chunk.contiguous(), right)

        # receive chunk
        src_chunk_idx = (dst_chunk_idx - 1) % W
        # print(torch.empty_like(t_chunks[src_chunk_idx]))
        buffer = torch.zeros_like(t_chunks[src_chunk_idx])
        dist.recv(buffer, left)
        t_chunks[src_chunk_idx] += buffer
        req.wait()
        # if src_chunk_idx == 0:
        #     print("Rank:", r, "sends", dst_chunk_idx, "and received chunk", src_chunk_idx, "and updated it to", t_chunks[src_chunk_idx])



    # second phase is the all-scatter which just sends the chunk they just recieved one further
    # the chunk we own is r
    # so we still have to use left and right here and just iterate W-1 over the items. From the element we have we go backwards.
    for i in range(W-1):
        send_idx = (r-i) % W
        recv_idx = (r-i-1) % W
        req = dist.isend(t_chunks[send_idx].contiguous(), right)
        
        buffer = torch.empty_like(t_chunks[recv_idx])
        dist.recv(buffer, left)
        req.wait()
        t_chunks[recv_idx].copy_(buffer)

    return t
                




# ---------------------------------------------------------------------------
# Part 2: manual forward/backward for  loss = mean((relu(x @ W1) @ W2 - y)^2)
# ---------------------------------------------------------------------------
def forward_backward(x, y, W1, W2, n_global):
    """Returns (loss, dW1, dW2) WITHOUT autograd.
    Normalise the loss by the GLOBAL batch size (n_global * out_dim), not the local
    shard, so the sum over ranks equals the full-batch gradient. Why does that matter?
    Chain: h_pre = x@W1 -> h = relu(h_pre) -> out = h@W2 -> loss
    """
    # Forward Pass
    # x (N, D)
    h_pre = x @ W1                    # (N, D) @ (D, H) = (N, H)
    h = torch.relu(h_pre)             # (N, H)
    out = h @ W2                      # (N, H) @ (H, O) = (N, O)
    diff = out - y 
    loss = (diff ** 2).sum() / (n_global * y.shape[1])     # loss: (N, O)
    # loss = ((torch.relu(x @ a1) @ a2 - y) ** 2).mean()
    
    d_out = 2 * diff / (n_global * y.shape[1])                        # dL/dout or dL/ddiff
    d_W2 =  h.T @ d_out                      # (N, H).T @ (N, O) = (H, O)       # dL/dW2
    d_h = d_out @ W2.T                       # (N, O) @ (H, O).T = (N, H)       # dL/dh
    # dh_pre = torch.where(h_pre > 0, d_h, 0)  # (N, H)                           # dL/dh_pre
    dh_pre = d_h * (0 < h_pre)  # (N, H)                           # dL/dh_pre
    d_W1 = x.T @ dh_pre                      # (D, H)                           # dL/dW1

    return loss, d_W1, d_W2




# ---------------------------------------------------------------------------
# Part 3: data-parallel training step
# ---------------------------------------------------------------------------
def dp_step(x_shard, y_shard, W1, W2, n_global, lr):
    """Local fwd/bwd -> all-reduce grads with YOUR ring_all_reduce -> SGD in place.
    Return the global loss (also all-reduced) as a Python float.
    """
    lloss, lW1, lW2 = forward_backward(x_shard, y_shard, W1, W2, n_global)
    dW1 = ring_all_reduce(lW1)
    dW2 = ring_all_reduce(lW2)
    loss = ring_all_reduce(torch.tensor([lloss.item()]))
    W1 -= lr * dW1
    W2 -= lr * dW2
    return loss.item()    


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
    t = torch.randn(4, generator=torch.Generator().manual_seed(rank))
    expected = t.clone(); dist.all_reduce(expected)
    time.sleep(1)
    ring_all_reduce(t)
    assert torch.allclose(t, expected), f"rank {rank}: ring all-reduce mismatch"
    if rank == 0:
        print('Hurra deine all reduce collective communication ist RICHTIG!!!')

    # Test 2: manual backprop matches autograd
    x, y, W1, W2 = make_problem()
    _, dW1, dW2 = forward_backward(x, y, W1, W2, x.shape[0])
    a1, a2 = W1.clone().requires_grad_(), W2.clone().requires_grad_()
    
    time.sleep(1)
    loss = ((torch.relu(x @ a1) @ a2 - y) ** 2).mean()
    loss.backward()
    time.sleep(1)
    if rank == -1:
        print('a1', list(a1))
        print('w1', list(dW1))
        print('a2', list(a2))
        print('w2', list(dW2))
    assert torch.allclose(dW1, a1.grad) and torch.allclose(dW2, a2.grad), "backprop mismatch"
    if rank == 0:
        print('Hurra dein Forward Backward Pass ist RICHTIG!!!')

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
    mp.spawn(worker, args=(2, 0.1), nprocs=WORLD)
