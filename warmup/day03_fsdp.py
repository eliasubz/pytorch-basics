"""
Day 03 — EXERCISE (Ziel: ~40 min): FSDP from scratch.
Parameters are sharded; each layer is all-gathered just before use and freed right after.
Run:  python day03_fsdp.py
"""
import os, socket, math
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
torch.set_default_dtype(torch.float64)

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


def ring_reduce_scatter(flat):
    r, W = dist.get_rank(), dist.get_world_size()
    right, left = (r+1) % W, (r-1) % W
    chunks = list(torch.tensor_split(flat.clone(), W))    # Don't modify the input
    for step in range(W-1):
        # lets return  not r
        send_idx = (r-step-1) % W
        recv_idx = (r-step-2) % W
        req = dist.isend(chunks[send_idx], right)
        buf = torch.empty_like(chunks[recv_idx])
        dist.recv(buf, left)
        chunks[recv_idx] += buf
        req.wait()
    return chunks[r]
        


def ring_all_gather(my_chunk, out):

    rank, W = dist.get_rank(), dist.get_world_size()
    right, left = (rank + 1) % W, (rank - 1) % W
    chunks = torch.tensor_split(out, W)
    chunks[rank].copy_(my_chunk)

    for step in range(W-1):
        s, r = (rank-step) % W, (rank-step-1)
        req = dist.isend(chunks[s].contiguous(), right)
        buf = torch.empty_like(chunks[r])
        dist.recv(buf, left)
        chunks[r].copy_(buf)
        req.wait()

    return out



# Counts how many parameter elements are currently gathered (full) on this rank.
class Mem:
    now = 0
    peak = 0

    @staticmethod
    def alloc(n):
        Mem.now += n; Mem.peak = max(Mem.peak, Mem.now)

    @staticmethod
    def release(n):
        Mem.now -= n


# Part 1: a sharded parameter
class FSDPParam:
    """One weight matrix, sharded across ranks. Each rank keeps only 1/W permanently.

    __init__(full_weight):
        - remember shape and numel
        - pad the flattened weight with zeros up to a multiple of W (self.padded),
          so every rank gets an equally sized shard (math.ceil helps)
        - self.shard = your 1/W piece, as its OWN memory (.clone(), why?)
        - self.full = None, self.grad_shard = None
    gather():
        - all-gather the shards into a padded flat tensor
        - self.full = the first numel elements, viewed back to the original shape
        - call Mem.alloc(self.padded) and return self.full
    free():
        - drop the full weight (self.full = None) and call Mem.release(self.padded)
    reduce_scatter_grad(full_grad):
        - pad the flat grad the same way, reduce-scatter -> self.grad_shard
    """
    def __init__(self, full_weight):
        W, r = dist.get_world_size(), dist.get_rank()
        self.w = W
        self.fw_shape = full_weight.shape
        self.fw_numel = full_weight.numel()
        self.padded = W*math.ceil(self.fw_numel/W)
        self.fw_padded = torch.zeros(self.padded)
        self.fw_padded[:self.fw_numel] = full_weight.reshape(-1)
        self.shard = torch.tensor_split(self.fw_padded, W)[r].clone()
        self.full = None
        self.grad_shard = None

    
    def gather(self):
        if self.full is not None:          # already gathered -> no new communication, no new memory
            return self.full
        self.fw_padded = ring_all_gather(self.shard, self.fw_padded)
        self.full = self.fw_padded[:self.fw_numel].reshape(self.fw_shape)
        Mem.alloc(self.padded) # Why works alloc like this?
        return self.full
        

    def free(self):
        if self.full is None:              # nothing to free
            return
        self.full = None
        Mem.release(self.padded)

    def reduce_scatter_grad(self, full_grad):
        """flattens and pads the gradients before reduce-scatter collective"""
        buf = torch.zeros(self.fw_padded.shape)
        buf[:self.fw_numel] = full_grad.reshape(-1)
        self.grad_shard = ring_reduce_scatter(buf)

# symptom P1 gather == w1 but P2 gather not equal W2
        

# ---------------------------------------------------------------------------
# Part 2: one FSDP training step for  loss = mean((relu(x @ W1) @ W2 - y)^2)
# ---------------------------------------------------------------------------
def fsdp_step(x, y, P1, P2, n_global, lr):
    """One training step. The rule: a layer is gathered ONLY while it is needed, and
    at most one layer may be gathered at any moment (Test 3 checks this).

    Forward:  P1 gather -> h_pre = x @ W1 -> free  ->  relu  ->  P2 gather -> out -> free
    Loss:     normalised by the GLOBAL size (n_global * out_dim), like Day 1
    Backward (reverse order):
        - which layer needs its weight again? gather it, compute, free it
        - after computing a layer's grad: P.reduce_scatter_grad(dW)
        - think: does dW1 need W1? Does dW2 need W2? Does d_h need W2?
    Optimizer: plain SGD on your shard only:  P.shard -= lr * P.grad_shard
    Return: the global loss as a float (dist.all_reduce is fine here).

    Questions:
     - In the forward pass we freed W2, then gathered it again in backward. What does that
       cost in communication, and what would it cost in memory not to? (FSDP calls this
       reshard_after_forward.)
       coms: R*mem(W2-1)* 2, it would cost mem(W2) throughout forward and backward, which hurts if the model is too big for all gpus to hold all params.
       shard after forward makes a lot of sense as name
     - Real FSDP prefetches the NEXT layer's all-gather while the current layer computes. Why?
       Answer: Because that way compuation and all-gather can amortize the time and therefor it is faster.
    """
    # only one full layer at a time with mem.peak 
    # loss = mean((relu(x @ W1) @ W2 - y)^2)
    # for matmul I need the full mats if I dont implement tiled matmuls, duhu
    W1 = P1.gather()
    h_pre = x @ W1
    P1.free()
    h = torch.clamp_min(h_pre, 0)

    W2 = P2.gather()
    o = h @ W2      # [s, o] = [s, h] @ (h, O)
    P2.free()

    diff = o - y
    loss = (diff ** 2).sum() / (n_global * y.shape[1])

    loss = ring_all_reduce(loss)

    # 16 min loss is correct
    # BW-pass
    d_out = (2/(n_global * y.shape[1])) * diff # dL/d_out = 2e * coef; gives local d_out
    # (H, O) = (H, s) @ (s, O)
    d_w2 = h.T @ d_out           # dL/dout * dout/dw2
    d_h = d_out @ P2.gather().T
    P2.free()
    P2.reduce_scatter_grad(d_w2)

    d_hpre = d_h * (0 < h_pre)
    d_w1 = x.T @ d_hpre    # [D, H]
    P1.reduce_scatter_grad(d_w1)

    r = dist.get_rank()

    P1.shard -= lr * P1.grad_shard
    P2.shard -= lr * P2.grad_shard

    return loss

# Mistakes 
# - not using grad_scatter
# - using mean for loss but with different sized shards this gives a wrong result
# - calling reduce_sccatter on unflattened array, because of tensor_split working on dim=0






# ---------------------------------------------------------------------------
# Harness (don't edit)
# ---------------------------------------------------------------------------
D, H, O, N = 2, 2, 2, 12      # odd sizes -> W1 has 91, W2 has 39 elements -> padding needed


def make_problem():
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


def worker(rank, port, steps, lr):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, timeout=timedelta(seconds=30))
    torch.set_default_dtype(torch.float64)
    x, y, W1, W2 = make_problem()

    # Test 1: sharding + gather round-trip
    P1 = FSDPParam(W1)
    import time
    time.sleep(1)
    P2 = FSDPParam(W2)
    assert P1.shard.numel() == math.ceil(W1.numel() / WORLD), "shard has the wrong size"
    if rank == 0:
        print("Shard has the right size")
    assert P1.full is None, "nothing should be gathered right after __init__"
    if rank == 0: print("nothing is gathered after __init__ ")
    assert torch.equal(P1.gather(), W1) and P1.full.shape == W1.shape, "gather != original"
    if rank == 0: print("Gather == original good")
    P1.free()
    assert P1.full is None and Mem.now == 0, "free() must drop the full weight and call Mem.release"

    if rank == 0: print("Test 1 passed: shard + gather + free")

    # Test 2: FSDP training == single-process training
    Mem.peak = 0
    s = N // WORLD
    steps = 1
    xs, ys = x[rank * s:(rank + 1) * s], y[rank * s:(rank + 1) * s]
    losses = [fsdp_step(xs, ys, P1, P2, N, lr) for _ in range(steps)]
    train_peak = Mem.peak
    assert P1.full is None and P2.full is None, "fsdp_step must free every layer it gathered"
    ref_losses, rW1, rW2 = reference(steps, lr)
    
    # print("gathered P2", P2.gather())
    # print("rw2", rW2)
    # print("gathered P1", P1.gather())
    # print("rw1", rW1)


    # print("W2", W2)
    # print("rfloss", ref_losses)
    # print("differnce", (W2 - rW2))

    time.sleep(2)
    assert all(abs(a - b) < 1e-9 for a, b in zip(losses, ref_losses)), "loss mismatch"
    assert torch.allclose(P1.gather(), rW1) and torch.allclose(P2.gather(), rW2), "weights diverged"
    P1.free(); P2.free()
    if rank == 0: print("Test 2 passed: FSDP == full-batch training")

    # Test 3: memory. At most ONE layer may be gathered at any moment.
    biggest = max(P1.padded, P2.padded)
    assert train_peak <= biggest, f"peak gathered params {train_peak} > largest layer {biggest}: free earlier!"
    if rank == 0:
        total = P1.padded + P2.padded
        print(f"Test 3 passed: peak gathered {train_peak} of {total} params, "
              f"persistent shard {P1.shard.numel() + P2.shard.numel()} per rank")
        print(f"All tests passed. loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    dist.destroy_process_group()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    mp.spawn(worker, args=(free_port(), 20, 0.1), nprocs=WORLD)
