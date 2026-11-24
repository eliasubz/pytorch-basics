"""
Day 05 — EXERCISE (target: ~40 min): Pipeline parallelism (GPipe schedule) from scratch.
Run:  python logrun.py day05_pp.py

DP: split the data.  FSDP: split the weights' memory.  TP: split each matrix.
PP (today): split the MODEL BY LAYERS. Rank r holds layer r only; activations flow
            rank 0 -> 1 -> 2 -> 3 in the forward pass and gradients flow back 3 -> 2 -> 1 -> 0.

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
# Part 1: one pipeline stage = one linear layer (+ ReLU, except the last stage)
# ---------------------------------------------------------------------------
class Stage:
    """One layer: z = x @ W, then ReLU (except on the last stage).

    __init__(W, is_last): store W and is_last; self.saved = {} ; self.dW = zeros like W
    forward(x, mb):  compute the output. Save what backward will need, keyed by the
                     microbatch id mb (several microbatches are "in flight" at once!).
    backward(d_out, mb): take the saved tensors for mb OUT of self.saved (Test 1 checks they
                     are gone afterwards), ACCUMULATE the weight grad into self.dW (+=, why?),
                     return the gradient w.r.t. the input x for the previous stage.
    """
    def __init__(self, W, is_last=False):
        self.W = W
        self.is_last = is_last
        self.saved = {}
        self.dW = torch.zeros_like(W)

    def forward(self, x, mb):
        # print(dist.get_rank(), "and input is", x)
        z = x @ self.W
        self.saved[mb] = (x, z)
        return z if self.is_last else z.clamp_min(0)

    def backward(self, d_out, mb):
        # x @ W = z
        # relu(z) = out
        x, z = self.saved.pop(mb)
        dz = d_out if self.is_last else d_out * (0 < z)     # dL/dz
        self.dW += x.T @ dz                                 # dL/dW = dz/dW * dL/dz
        dx = dz @ self.W.T
        return dx


# ---------------------------------------------------------------------------
# Part 2: GPipe step: all forwards, then all backwards, then one optimizer step
# ---------------------------------------------------------------------------
def gpipe_step(stage, x_mbs, y_mbs, shapes, n_global, lr):
    """GPipe: forward ALL microbatches, then backward ALL microbatches, then one SGD step.

    rank 0 has the inputs:   x_mbs = list of M microbatches (None on other ranks)
    last rank has targets:   y_mbs = list of M microbatches (None on other ranks)
    shapes[mb] = number of rows of microbatch mb (so you can allocate recv buffers)

    Forward, for each mb:
        input  = x_mbs[mb] on rank 0, otherwise dist.recv from rank-1
        output = stage.forward(...)
        last rank: compute this mb's loss and d_out (normalise by n_global * out_dim!),
                   keep d_out until the backward pass
        other ranks: dist.send the output to rank+1
    Backward, for each mb in REVERSE order:
        d_out = the stored one on the last rank, otherwise dist.recv from rank+1
        d_in  = stage.backward(...)
        if not rank 0: dist.send d_in to rank-1
    Then: SGD on stage.W with stage.dW, and reset dW to zero.
    Return the total loss (float) on the last rank, None elsewhere.

    Buffer shapes: what is the width of the activation arriving at this stage? Of the
    gradient arriving from the next stage? (Look at stage.W.shape.)

    Questions (look at the timeline the harness prints):
     - What fraction of the time is each rank idle? Derive (S-1)/(M+S-1).
       What happens with M=1? With M=32?
     - How many microbatch activations does rank 0 hold at most? Why is that a problem,
       and how does the 1F1B schedule (one forward, one backward, alternating) fix it?
     - Plain blocking send/recv worked here. Why does it NOT deadlock, unlike Day 1's ring?
    """
    # list([N, D])
    r, W = dist.get_rank(), dist.get_world_size()
    first, last = r == 0, r == W - 1
    out = 0
    loss = 0
    d_outs = {}

    for mb in range(len(shapes)):

        # forward 
        if r == 0:
            x = x_mbs[mb]
            # print("mb", mb, "and x", x)
        else:
            x = torch.empty(shapes[mb], stage.W.shape[0])
            dist.recv(x, r-1)

        z = stage.forward(x, mb)
        if last:
            diff = z - y_mbs[mb]
            loss += (diff ** 2).sum() / (n_global * diff.shape[1])
            d_out = (2 * diff) / (n_global * stage.W.shape[-1])
            d_outs[mb] = d_out
        if r < W-1:
            # print("sending z to", r+1)
            # print(z)
            dist.send(z, r+1)

    for mb in reversed(range(len(shapes))):     
        # loss and BW
        if last:
            d_out = d_outs[mb]
        else:
            d_out = torch.empty((shapes[mb], stage.W.shape[-1]))
            dist.recv(d_out, r+1)
        
        d_in = stage.backward(d_out, mb)
        if not first:
            dist.send(d_in, r-1)
        

    stage.W -= lr * stage.dW
    stage.dW.zero_()

    return loss if stage.is_last else None

    

    
            
            
    
    


# ---------------------------------------------------------------------------
# Harness (don't edit)
# ---------------------------------------------------------------------------
D, O, N, M = 8, 3, 32, 4          # 4 layers 8->8->8->8->3, batch 32 in 4 microbatches of 8
# (the timeline below assumes every stage call takes the same time)


def make_problem():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, D, generator=g)
    y = torch.randn(N, O, generator=g)
    Ws = [torch.randn(D, D, generator=g) * 0.5 for _ in range(WORLD - 1)]
    Ws.append(torch.randn(D, O, generator=g) * 0.5)
    return x, y, Ws


def reference(steps, lr):
    x, y, Ws = make_problem()
    Ws = [w.requires_grad_() for w in Ws]
    losses = []
    for _ in range(steps):
        h = x
        for i, w in enumerate(Ws):
            h = h @ w
            if i < len(Ws) - 1:
                h = torch.relu(h)
        loss = ((h - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            for w in Ws:
                w -= lr * w.grad; w.grad = None
        losses.append(loss.item())
    return losses, [w.detach() for w in Ws]


def instrument(stage, trace):
    """Wraps forward/backward to record the ORDER in which this rank runs them."""
    fwd, bwd = stage.forward, stage.backward
    def rec(fn, tag):
        def wrapper(t, mb):
            trace.append(f"{tag}{mb}"); return fn(t, mb)
        return wrapper
    stage.forward, stage.backward = rec(fwd, "F"), rec(bwd, "B")


def print_timeline(traces):
    """Time slot of an event = 1 + max(previous event on this rank, the event it waits for):
    F_mb on rank r waits for F_mb on rank r-1, B_mb on rank r waits for B_mb on rank r+1."""
    S = len(traces)
    slot, pos = {}, [0] * S
    while any(pos[r] < len(traces[r]) for r in range(S)):
        progressed = False
        for r in range(S):
            if pos[r] == len(traces[r]):
                continue
            ev = traces[r][pos[r]]
            dep = (r - 1, ev) if ev[0] == "F" and r > 0 else (r + 1, ev) if ev[0] == "B" and r < S - 1 else None
            if dep and dep not in slot:
                continue
            prev = slot[(r, traces[r][pos[r] - 1])] if pos[r] else -1
            slot[(r, ev)] = max(prev, slot[dep] if dep else -1) + 1
            pos[r] += 1; progressed = True
        assert progressed, "schedule deadlocks"
    width = max(slot.values()) + 1
    print("\nTimeline (one column = one time slot, '..' = idle = pipeline bubble):")
    for r in range(S):
        row = {slot[(r, ev)]: ev for ev in traces[r]}
        print(f"  rank {r}: " + " ".join(row.get(t, "..") for t in range(width)))
    busy = sum(len(t) for t in traces)
    print(f"  busy {busy} of {width * S} slots -> bubble {1 - busy / (width * S):.0%}")


def worker(rank, port, steps, lr):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, timeout=timedelta(seconds=60))
    torch.set_default_dtype(torch.float64)
    x, y, Ws = make_problem()
    last = rank == WORLD - 1

    # Test 1: a single stage, forward + backward against autograd
    W_test = Ws[0].clone()
    xt = torch.randn(5, D, generator=torch.Generator().manual_seed(1))
    st = Stage(W_test.clone(), is_last=False)
    out = st.forward(xt, mb=0)
    g_out = torch.randn_like(out)
    d_in = st.backward(g_out, mb=0)
    xa, Wa = xt.clone().requires_grad_(), W_test.clone().requires_grad_()
    torch.relu(xa @ Wa).backward(g_out)
    assert torch.allclose(out, torch.relu(xt @ W_test)), "stage forward wrong"
    assert torch.allclose(d_in, xa.grad), "stage backward: input gradient wrong"
    assert torch.allclose(st.dW, Wa.grad), "stage backward: weight gradient wrong"
    assert len(st.saved) == 0, "backward should free the saved activations (pop them)"
    if rank == 0: print("Test 1 passed: one stage forward + backward")

    # Test 2: pipeline training == single-process training
    stage = Stage(Ws[rank].clone(), is_last=last)
    mb_size = N // M
    shapes = [mb_size] * M
    x_mbs = list(x.split(mb_size)) if rank == 0 else None
    y_mbs = list(y.split(mb_size)) if last else None
    losses = [gpipe_step(stage, x_mbs, y_mbs, shapes, N, lr) for _ in range(steps)]
    ref_losses, ref_Ws = reference(steps, lr)
    assert torch.allclose(stage.W, ref_Ws[rank]), f"rank {rank}: weights diverged"
    if last:
        assert all(abs(a - b) < 1e-9 for a, b in zip(losses, ref_losses)), "loss mismatch"
    dist.barrier()
    if rank == 0: print("Test 2 passed: GPipe == single-process training")

    # Timeline: one more step, recording the order of stage calls, to SEE the bubble
    trace = []
    instrument(stage, trace)
    gpipe_step(stage, x_mbs, y_mbs, shapes, N, lr)
    traces = [None] * WORLD
    dist.all_gather_object(traces, trace)
    if rank == 0:
        print_timeline(traces)
        print(f"  theory: bubble = (S-1)/(M+S-1) = {(WORLD-1)/(M+WORLD-1):.0%} with S={WORLD} stages, M={M} microbatches")
    dist.destroy_process_group()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    mp.spawn(worker, args=(free_port(), 10, 0.05), nprocs=WORLD)
