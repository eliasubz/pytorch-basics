"""
Ring all-reduce from scratch, compared against the naive version and torch's built-in one.
Run:  python ring_allreduce.py
"""
import os, socket, time
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def naive_all_reduce(x):
    """Your version from today: everyone -> rank 0 -> everyone.
    Rank 0 receives and sends (W-1) * N elements, so it becomes the bottleneck."""
    rank, W = dist.get_rank(), dist.get_world_size()
    if rank == 0:
        buf = torch.empty_like(x)
        for src in range(1, W):
            dist.recv(buf, src)
            x += buf
        for dst in range(1, W):
            dist.send(x, dst)
    else:
        dist.send(x, 0)
        dist.recv(x, 0)
    return x


def ring_all_reduce(x):
    """Bandwidth-optimal all-reduce. Every rank sends only 2 * (W-1)/W * N elements,
    which is almost independent of W, and all links are busy at the same time.

    Ranks sit in a circle: each sends to its right neighbour and receives from its left.
    The tensor is cut into W chunks.
      Phase 1, reduce-scatter (W-1 steps): pass chunks around and ADD what you receive.
               Afterwards rank r holds the complete sum of chunk (r+1) % W.
      Phase 2, all-gather (W-1 steps): pass the finished chunks around and COPY them.
               Afterwards every rank holds every finished chunk.
    """
    rank, W = dist.get_rank(), dist.get_world_size()
    right, left = (rank + 1) % W, (rank - 1) % W

    # Views into x (no copies!): writing into chunks[i] writes into x itself.
    chunks = x.view(-1).tensor_split(W)
    recv_buf = torch.empty_like(chunks[0])  # chunk 0 is the largest when sizes are uneven

    def send_recv(send_idx, recv_idx):
        buf = recv_buf[: chunks[recv_idx].numel()]
        # isend doesn't block, so every rank can send AND receive at the same time.
        # If all ranks used a blocking send first, they could all wait on each other forever.
        req = dist.isend(chunks[send_idx], right)
        dist.recv(buf, left)
        req.wait()
        return buf

    # Phase 1: reduce-scatter
    for step in range(W - 1):
        send_idx = (rank - step) % W
        recv_idx = (rank - step - 1) % W
        chunks[recv_idx].add_(send_recv(send_idx, recv_idx))

    # Phase 2: all-gather
    for step in range(W - 1):
        send_idx = (rank - step + 1) % W
        recv_idx = (rank - step) % W
        chunks[recv_idx].copy_(send_recv(send_idx, recv_idx))

    return x


def bench(fn, x, iters=5):
    times = []
    for _ in range(iters):
        y = x.clone()
        dist.barrier()                 # start all ranks together
        t0 = time.perf_counter()
        fn(y)
        dist.barrier()
        times.append(time.perf_counter() - t0)
    return min(times) * 1000, y


def worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=60))

    # Correctness check: odd size, so chunks are uneven
    x = torch.randn(1001, generator=torch.Generator().manual_seed(rank))
    expected = x.clone(); dist.all_reduce(expected)
    assert torch.allclose(ring_all_reduce(x.clone()), expected, atol=1e-5)
    assert torch.allclose(naive_all_reduce(x.clone()), expected, atol=1e-5)

    # Speed: 16M floats = 64 MB, roughly the gradients of a 16M-parameter model
    big = torch.randn(16_000_000)
    results = {name: bench(fn, big)[0] for name, fn in
               [("naive", naive_all_reduce), ("ring", ring_all_reduce),
                ("torch", lambda t: dist.all_reduce(t))]}
    if rank == 0:
        print(f"correct on {world_size} ranks")
        for name, ms in results.items():
            print(f"  {name:6s} {ms:8.1f} ms")
    dist.destroy_process_group()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    W = 4
    mp.spawn(worker, args=(W, free_port()), nprocs=W)