"""
Day 06 — EXERCISE (target: ~45 min): Expert parallelism (Mixture of Experts) with all-to-all from scratch.
Run:  python logrun.py day06_ep.py

Setup: 4 ranks, 4 experts, expert e lives ONLY on rank e. Every rank has its own tokens.
A router picks one expert per token (top-1). Tokens travel to their expert's rank (all-to-all),
get processed there, and travel back (all-to-all again). Backward does the same in reverse.

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
# Part 1: all-to-all with equal sizes
# ---------------------------------------------------------------------------
def all_to_all_equal(send):
    """send: list of W tensors, all with the same shape. send[j] goes to rank j.
    Return recv: list of W tensors, recv[j] = what rank j sent to me.
    - The piece for yourself (send[rank]) needs no network: just copy it.
    - For every other rank: isend AND irecv (irecv = non-blocking recv), then wait for all.
    Why is blocking send/recv in a loop over j dangerous here? (Think: Day 1's hint.)
    """
    r, W = dist.get_rank(), dist.get_world_size()
    recv = [torch.empty_like(send[0]) for _ in range(W)]
    recv[r].copy_(send[r])
    reqs = []

    for j in range(W):
        if j == r:
            continue
        reqs.append(dist.isend(send[j], j))
        reqs.append(dist.irecv(recv[j], j))
    for req in reqs:
        req.wait()

    return recv



# ---------------------------------------------------------------------------
# Part 2: all-to-all with variable sizes (rows differ per destination)
# ---------------------------------------------------------------------------
def all_to_all_var(send):
    """Like all_to_all_equal, but send[j] has shape (n_j, *feat) and n_j differs per j
    (and can be 0). Problem: the receiver doesn't know how many rows will arrive.
    Step 1: tell everyone how many rows you'll send them. That's an all-to-all of
            1-element tensors, which all have the same shape -> use all_to_all_equal.
    Step 2: allocate recv buffers with the right sizes, then exchange the data.
    Skip empty messages, on BOTH sides consistently (why must both sides agree?).
    """
    r, W = dist.get_rank(), dist.get_world_size()
    
    # step 1 explain
    my_shapes = [torch.tensor(int(item.shape[0])) for item in send]
    their_shapes = all_to_all_equal(my_shapes)

    # step 2 build recieve buffer and send data
    recv = [torch.empty((rows.item(), send[0].shape[-1])) for rows in their_shapes]
    recv[r].copy_(send[r])
    reqs = []
    for j in range(W):
        if j == r:
            continue
        reqs.append(dist.isend(send[j], j))
        reqs.append(dist.irecv(recv[j], j))
    for req in reqs:
        req.wait()

    return recv   
# lets go! It took only 9 minutes for this with only empty_like and rows dtype erros     


# ---------------------------------------------------------------------------
# Part 3: dispatch (sort tokens by expert) and its inverse
# ---------------------------------------------------------------------------
def dispatch(x, expert_idx, E):
    """Group my tokens by expert.
    x: (n, D), expert_idx: (n,) with values in 0..E-1
    Return (send, order, counts):
      send[e]  = the rows of x routed to expert e (keep their original relative order)
      order    = the permutation you applied (you need it to undo the sorting later)
      counts   = number of tokens per expert, length E
    Useful: torch.argsort(..., stable=True), torch.bincount(..., minlength=E), tensor.split(list)
    """
    # so we get n tokens that are send to n different experts, which we have E of. 
    order = torch.argsort(expert_idx, stable=True)
    counts = torch.bincount(expert_idx, minlength=E)
    # Ich brauche die liste an größen aber gleichzeitig müssen die gruppen auch richtig sein, wenn ein expert leer ist
    send = list(x[order].split(counts.tolist()))
    return send, order, counts
    

def undispatch(chunks, order):
    """Inverse of dispatch: chunks are in expert order -> put every row back to its
    original position. Hint: if y_sorted = y[order], then y[order] = y_sorted.
    """
    y_sorted = torch.cat(chunks, dim=0)
    y = torch.empty_like(y_sorted)
    y[order] = y_sorted
    return y



# ---------------------------------------------------------------------------
# Part 4: one EP training step
# ---------------------------------------------------------------------------
def moe_step(x, y, W_expert, W_router, n_global, lr):
    """One step. Every rank: its own tokens x, y. W_expert = the ONE expert on this rank.

    Route:    probs = softmax(x @ W_router); gate, expert_idx = top-1 (probs.max(dim=-1))
              The router is NOT trained today: treat gate as a constant.
    Forward:  dispatch -> all_to_all_var -> concat what you received = tokens for MY expert
              -> h = relu(x_e @ W_expert)
              -> send each rank's results back (all_to_all_var again: what sizes?)
              -> undispatch -> out = gate * h  (per token)
    Loss:     (diff**2).sum() / (n_global * D), all-reduced (dist.all_reduce is fine)
    Backward: d_out -> d_h for my tokens -> send the grads to the experts (same path as
              the tokens went!) -> d_z -> dW = x_e.T @ d_z
    Update:   W_expert -= lr * dW  (in place)
    Return:   (global loss as float, number of tokens my expert processed)

    Questions:
     - Unlike DP, the expert's weight gradient needs no all-reduce. Why? When WOULD it
       need one? (Hint: real systems combine EP with DP.)
     - Look at "Tokens per expert" at the end. Expert 2 gets 12 tokens, expert 0 only 5.
       What does that mean for the speed of the whole step? What are "capacity factor"
       and "load balancing loss" for?
     - How many all-to-alls per step (forward + backward)? Why can't they overlap with
       the expert's compute as easily as FSDP's all-gather could?
     - Keep in mind that the moe step is not using attention thats why we dont need any communication
       during the forward step. The reason that the gradients also dont need comms is that the paths of 
       different experts do not align. If we look at their flows we see that each token uses a different
       expert and therefore only has differential impact on these weights on exactly these experts. On the other hand
       if every x passes through top-n best experts, the thing looks the same because expert i does not care that 
       data point x passed through another expert. It would need all readuce if we combined DP and EP and expert i 
       would run on multiple ranks with different data points then the gradients would need to all reduce. 
    """
    xs = x @ W_router               # [s, W]
    probs = torch.softmax(xs, dim=-1)   # [s, W]
    gate, expert_idx =  torch.max(probs, dim=1) # [s,]   # returns (val, idx)
    
    send, order, counts = dispatch(x, expert_idx, W_router.shape[1])   
    recv = all_to_all_var(send)                     # list(E, [sj, D])
    x_e = torch.cat(recv)                           # [local_n, D]
    h_pre = x_e @ W_expert                          # [local_n, D]
    h = h_pre.clamp_min(0)                          # [local_n, D]
    back = all_to_all_var(list(torch.split(h, [rec.shape[0] for rec in recv])))  # remember that recv[j] is what rank j sent us
    h_local = undispatch(back, order)               # [s, D]
    out = gate[:, None] * h_local              # [s, D]

    # loss
    denom = n_global * out.shape[1]            # scalar
    diff = out - y                             # [s, D]
    loss = torch.tensor([(diff ** 2).sum().item() / denom])
    dist.all_reduce(loss)           # every rank holds a different loss because of our denom we can sum it 

    # BW
    d_out = 2 * diff / denom    # dL / d_out
    d_h_local = d_out * gate[:, None]
    d_send = list(d_h_local[order].split(counts.tolist()))
    dh = torch.cat(all_to_all_var(d_send))
    dz = dh * (h_pre > 0)
    dWe = x_e.T @ dz
    d_x_e = dz @ W_expert

    d_gate = (d_out * h_local).sum(dim=-1)      # [s, D]   dL/d_out * d_out/d_gate
    
    # because after softmax we do max the probs bw is just softmax 
    onehot = torch.nn.functional.one_hot(expert_idx, W_router.shape[1]).to(x.dtype) # [s, E]
    # crazy stuff
    d_logits = d_gate[:, None] * probs * (onehot - gate[:, None])
    d_W_router = x.T @ d_logits

    dist.all_reduce(d_W_router)

    W_router -= lr * d_W_router
    W_expert -= lr * dWe
    
    return loss.item(), x_e.shape[0]

# I understand d_softmax = (dp - (dp * p)).sum(dim=-1, keepdim=True) but at the same time I dont understand it
# forward pass makes sens although my head doesnt fly through the intuition of how undispatch reordering works
# but I understand its in- and outputs. The collectives in the moe step make sense but also are still a bit foreign to me.
# its also funny to me that d_x_e is never used but it makes sense because most of the tokens in there come from other ranks.
    





# ---------------------------------------------------------------------------
# Harness (don't edit)
# ---------------------------------------------------------------------------
D, N = 6, 32                      # 32 tokens in total, 8 per rank, width 6


def make_problem():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, D, generator=g)
    y = torch.randn(N, D, generator=g)
    experts = [torch.randn(D, D, generator=g) * 0.4 for _ in range(WORLD)]
    W_router = torch.randn(D, WORLD, generator=g)
    return x, y, experts, W_router


def reference(steps, lr):
    x, y, experts, W_router = make_problem()
    experts = [e.requires_grad_() for e in experts]
    W_router.requires_grad_()
    losses, first_idx = [], None
    for _ in range(steps):
        # routing must be recomputed every step: the router changes now
        probs = torch.softmax(x @ W_router, dim=-1)
        gate, idx = probs.max(dim=-1)
        if first_idx is None:
            first_idx = idx
        out = torch.stack([gate[i] * torch.relu(x[i] @ experts[idx[i]]) for i in range(N)])
        loss = ((out - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            for e in experts:
                e -= lr * e.grad; e.grad = None
            W_router -= lr * W_router.grad; W_router.grad = None
        losses.append(loss.item())
    return losses, [e.detach() for e in experts], torch.bincount(first_idx, minlength=WORLD), W_router.detach()


def worker(rank, port, steps, lr):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, timeout=timedelta(seconds=30))
    torch.set_default_dtype(torch.float64)
    W = WORLD

    # Test 1: equal-size all-to-all. Rank i sends the value 10*i + j to rank j.
    send = [torch.full((2, 3), 10.0 * rank + j) for j in range(W)]
    recv = all_to_all_equal(send)
    for j in range(W):
        assert torch.equal(recv[j], torch.full((2, 3), 10.0 * j + rank)), f"rank {rank}: wrong data from rank {j}"
    if rank == 0: print("Test 1 passed: all-to-all (equal sizes)")

    # Test 2: variable sizes, including empty messages
    rows = lambda i, j: (i + 2 * j) % 4                  # 0..3 rows, some are 0
    send = [torch.full((rows(rank, j), 3), 10.0 * rank + j) for j in range(W)]
    recv = all_to_all_var(send)
    for j in range(W):
        assert torch.equal(recv[j], torch.full((rows(j, rank), 3), 10.0 * j + rank)), \
            f"rank {rank}: wrong data or size from rank {j}"
    if rank == 0: print("Test 2 passed: all-to-all (variable sizes, incl. empty)")

    # Test 3: dispatch / undispatch round trip (local, no comms)
    xt = torch.randn(10, D, generator=torch.Generator().manual_seed(rank))
    idx = torch.tensor([3, 0, 1, 3, 3, 0, 2, 1, 0, 3])
    s, order, counts = dispatch(xt, idx, W)
    assert [c.shape[0] for c in s] == [3, 2, 1, 4], "dispatch: wrong number of tokens per expert"
    assert all(torch.equal(s[e], xt[idx == e]) for e in range(W)), "dispatch: wrong tokens for an expert"
    assert torch.equal(undispatch(s, order), xt), "undispatch does not restore the original order"
    if rank == 0: print("Test 3 passed: dispatch + undispatch")

    # Test 4: EP training == single-process training
    x, y, experts, W_router = make_problem()
    s_ = N // W
    xs, ys = x[rank * s_:(rank + 1) * s_], y[rank * s_:(rank + 1) * s_]
    W_expert = experts[rank].clone()
    results = [moe_step(xs, ys, W_expert, W_router, N, lr) for _ in range(steps)]
    ref_losses, ref_experts, load, ref_router = reference(steps, lr)
    assert all(abs(a - b) < 1e-9 for (a, _), b in zip(results, ref_losses)), "loss mismatch"
    assert torch.allclose(W_expert, ref_experts[rank]), f"expert {rank} diverged"
    assert torch.allclose(W_router, ref_router), f"rank {rank}: router diverged"
    assert results[0][1] == load[rank], "this expert received the wrong number of tokens"
    if rank == 0:
        print("Test 4 passed: EP training == single-process training")
        print(f"All tests passed. loss {results[0][0]:.4f} -> {results[-1][0]:.4f}")
        print(f"Tokens per expert: {load.tolist()} (perfectly balanced would be {N // W} each)")
    dist.destroy_process_group()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]



if __name__ == "__main__":
    mp.spawn(worker, args=(free_port(), 10, 0.2), nprocs=WORLD)
