"""
Day 02 — EXERCISE (target: ~40 min): reduce-scatter + all-gather as separate collectives, then a ZeRO sharded optimizer.
Run:  python day02_zero.py
"""
import os, socket
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Part 1: ring reduce-scatter
# ---------------------------------------------------------------------------
def ring_reduce_scatter(flat: torch.Tensor) -> torch.Tensor:
    """Phase 1 of your ring all-reduce, as its own function.
    - Input: a 1-D tensor. Do NOT modify it (clone first). The test checks this.
    - Output: ONLY your fully summed chunk, i.e. sum over ranks of flat.tensor_split(W)[rank].
    You already wrote this loop yesterday: your version ended with rank r owning chunk r.
    """
    # t.view(-1)
    r, W = dist.get_rank(), dist.get_world_size()
    chunks = list(flat.clone().tensor_split(W))         # W chunks in a list
    left, right = (r-1) % W, (r+1) % W

    for i in range(W-1):
        send_idx = (r-i-1) % W
        recv_idx = (r-i-2) % W

        # send
        req = dist.isend(chunks[send_idx], right)
        buffer = torch.empty_like(chunks[recv_idx])

        # recieve
        dist.recv(buffer, left)
        chunks[recv_idx] += buffer
        req.wait()
    
    return chunks[r]

# Mistakes
# - forgot the adding


# ---------------------------------------------------------------------------
# Part 2: ring all-gather
# ---------------------------------------------------------------------------
def ring_all_gather(my_chunk: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Phase 2 of your ring all-reduce, as its own function.
    - my_chunk: this rank's piece (it belongs at position `rank`).
    - out: 1-D tensor to fill with ALL ranks' chunks, in rank order. Write into it in place.
    Steps: put my_chunk into its slot of out, then W-1 ring steps.
    Remember yesterday's bug: recv into a fresh buffer, then copy_ into the view.
    """
    r, w = dist.get_rank(), dist.get_world_size()
    left, right = (r-1) % w, (r+1) % w
    chunks = list(out.tensor_split(w))        # so this apparently gives views into out

    chunks[r].copy_(my_chunk)

    for i in range(w-1):
        send_idx = (r-i) % w
        recv_idx = (r-i-1) % w

        # send my complete chunk
        req = dist.isend(chunks[send_idx], right)

        # recieve copy
        buffer = torch.empty_like(chunks[recv_idx])
        dist.recv(buffer, left)
        chunks[recv_idx].copy_(buffer)

        req.wait()

    return out



# ---------------------------------------------------------------------------
# Part 3: ZeRO-1/2 style sharded SGD with momentum
# ---------------------------------------------------------------------------
class ShardedSGD:
    """ZeRO: each rank keeps optimizer state (momentum) for only 1/W of the parameters.

    SGD with momentum (same as torch.optim.SGD):   v = mu * v + g ;   p = p - lr * v

    __init__: store lr/momentum, take a VIEW of your shard of flat_params
              (tensor_split(W)[rank]) and create a momentum buffer the size of that shard only.
    
    step(flat_grad):
       1. reduce-scatter the grads  -> summed grad for your shard only
       2. update momentum + your param shard (in place, so flat_params changes)
       3. all-gather the param shards into flat_params, so every rank has all new params

    Questions to answer while you code:
     - Why reduce-scatter instead of all-reduce? Which bytes would all-reduce waste?
     - Total communication compared with yesterday's DP step?
     - With Adam (2 states per param) and W=64, how much optimizer memory does each rank save?
    """
    def __init__(self, flat_params, lr, momentum):
        self.r = dist.get_rank()
        self.W = dist.get_world_size()
        self.all_params = flat_params       # W1 and W2
        self.my_params_shard = torch.tensor_split(flat_params, self.W)[self.r]
        self.momentum_buf = torch.zeros_like(self.my_params_shard)
        self.lr = lr
        self.mu = momentum

    def step(self, flat_grad):

        # 1. get full gradient
        my_grad = ring_reduce_scatter(flat_grad)    # total sum of all gradients for shard r

        # 2. sgd gradient step
        # self.momentum_buf = self.mu * self.momentum_buf + my_grad
        self.momentum_buf.mul_(self.mu).add_(my_grad)

        # 3. all-gather parameters
        self.my_params_shard.sub_(self.lr * self.momentum_buf)
        ring_all_gather(self.my_params_shard.clone(), self.all_params)
# Am meisten hat gedauer herauszufindend dass meine daten leer waren, weil N=1 keine gescheiten shards hinbekmen hat weil s = N//W = 0 war
    



# ---------------------------------------------------------------------------
# Harness (don't edit)
# ---------------------------------------------------------------------------
D, H, O, N = 8, 1, 3, 5


def make_problem():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, D, generator=g)
    y = torch.randn(N, O, generator=g)
    W1 = torch.randn(D, H, generator=g) * 0.3
    W2 = torch.randn(H, O, generator=g) * 0.3
    return x, y, W1, W2


def unflatten(flat):
    """Views into the flat buffer -> no copies, updates to flat show up in W1/W2."""
    return flat[: D * H].view(D, H), flat[D * H:].view(H, O)


def grads(x, y, W1, W2, n_global):
    """Working manual backprop (it's yesterday's, done right). Returns flat grad."""


    h_pre = x @ W1
    h = h_pre.clamp_min(0)
    d_out = 2 * (h @ W2 - y) / (n_global * O)
    dW2 = h.T @ d_out
    dW1 = x.T @ ((d_out @ W2.T) * (h_pre > 0))
    out = torch.cat([dW1.reshape(-1), dW2.reshape(-1)])
    return out


def reference(steps, lr, mu):
    x, y, W1, W2 = make_problem()
    # params = grads(x, y, W1, W2, N)
    params = [W1.requires_grad_(), W2.requires_grad_()]
    opt = torch.optim.SGD(params, lr=lr, momentum=mu)
    for _ in range(steps):
        opt.zero_grad()
        ((torch.relu(x @ W1) @ W2 - y) ** 2).mean().backward()
        opt.step()
    return torch.cat([W1.detach().reshape(-1), W2.detach().reshape(-1)])


def worker(rank, port, steps, lr, mu):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, timeout=timedelta(seconds=30))
    torch.set_default_dtype(torch.float64)
    W = WORLD

    # Test 1: reduce-scatter. Odd size -> uneven chunks.
    x = torch.randn(21, generator=torch.Generator().manual_seed(rank))
    total = x.clone(); dist.all_reduce(total)
    x_before = x.clone()
    import time
    time.sleep(0.5)
    shard = ring_reduce_scatter(x)
    assert torch.allclose(shard, total.tensor_split(W)[rank]), f"rank {rank}: reduce-scatter wrong"
    assert torch.equal(x, x_before), "reduce-scatter must not modify its input"
    if rank == 0: print("Test 1 passed: reduce-scatter")

    # Test 2: all-gather
    out = torch.zeros(21)
    ring_all_gather(total.tensor_split(W)[rank].clone(), out)
    assert torch.allclose(out, total), f"rank {rank}: all-gather wrong"
    if rank == 0: print("Test 2 passed: all-gather")

    # Test 3: reduce-scatter + all-gather == all-reduce
    out = torch.zeros(21)
    ring_all_gather(ring_reduce_scatter(x), out)
    assert torch.allclose(out, total)
    if rank == 0: print("Test 3 passed: RS + AG == all-reduce")

    # Test 4: ZeRO training == single-process torch.optim.SGD with momentum
    xd, yd, W1, W2 = make_problem()
    flat = torch.cat([W1.reshape(-1), W2.reshape(-1)])
    W1, W2 = unflatten(flat)
    opt = ShardedSGD(flat, lr, mu)
    assert opt.momentum_buf.numel() <= flat.numel() // W + 1, "optimizer state is not sharded!"
    s = N // W

    xs, ys = xd[rank * s:(rank + 1) * s], yd[rank * s:(rank + 1) * s]


    for _ in range(steps):
        opt.step(grads(xs, ys, W1, W2, N))
    ref = reference(steps, lr, mu)
    assert torch.allclose(flat, ref), f"rank {rank}: ZeRO params diverged from reference"
    if rank == 0:
        print(f"Test 4 passed: ZeRO == torch SGD. Momentum state per rank: "
              f"{opt.momentum_buf.numel()} of {flat.numel()} params")
    dist.destroy_process_group()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    mp.spawn(worker, args=(free_port(), 1, 0.1, 0.9), nprocs=WORLD)
