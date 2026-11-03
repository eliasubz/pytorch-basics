import torch
from pathlib import Path

import time

def compile_kernel():
    kernel_path = (Path(__file__).parent / "matMul.metal").read_text()
    return torch.mps.compile_shader(kernel_path)

hi = compile_kernel()

def matMul(lib, x, y):
    assert x.device == y.device and x.device.type == "mps"
    assert x.dtype == y.dtype == torch.float
    assert x.size(-1) == y.size(-2)

    c = torch.empty(x.size(-2), y.size(-1), dtype=torch.float, device='mps')

    lib.matMulKernel(x, y, c, x.size(-2), x.size(-1), y.size(-1), threads=c.numel(), group_size=256)

    return c

def main():
    m = 16
    n = 32
    o = 64

    x = torch.randn((m, n), dtype=torch.float, device='mps')
    y = torch.randn((n, o), dtype=torch.float, device='mps')

    print('X and Y shape: (m,n) and (n,o)', x.shape, y.size())
    lib = compile_kernel()

    t0 = time.time()
    c = matMul(lib, x, y)
    torch.mps.synchronize()
    t1 = time.time()
    kernel_t = t1 - t0

    t0 = time.time()
    c_torch = x @ y
    torch.mps.synchronize()
    t1 = time.time()
    torch_t = t1 - t0


    print('Match:', torch.allclose(c, c_torch, rtol=1e-1))
    print('Torch time:', torch_t)
    print('Kernel time:', kernel_t)

if __name__ == "__main__":
    main()