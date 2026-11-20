import os, torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta


def worker(rank, world_size):
    # Every process runs this same function; only `rank` differs.
    os.environ["MASTER_ADDR"] = "127.0.0.1"   # where the processes meet
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30))  # gloo = CPU backend
    rank, world_size = dist.get_rank(), dist.get_world_size()
    r, W= rank, world_size

    x = torch.tensor([float(rank)])           # rank 0 holds [0.], rank 1 holds [1.], ...
    print(f"rank {rank} before: {x}")
    # every process sends its data to x and x sends its sum back

    if r == 0:
        # creating receving arr
        x_buf = torch.zeros(W-1,1) # (W-1,)
        print("x_buffer", x_buf.size())
        
        # receiving all vals
        for i, x_p in enumerate(x_buf):
            dist.recv(x_p, i+1)

        # calculating sum
        x += x_buf.sum()
        
        # scatter reduced value back out 
        for recv_r in range(1,W):
            dist.send(x, recv_r)
    else:
        # send value to rank 0
        dist.send(x, 0)
        # recv sum
        dist.recv(x, 0)

    # dist.all_reduce(x)                      # sums x across all processes, in place
    print(f"rank {rank} after:  {x}")

    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(4,), nprocs=4)     # start 4 processes