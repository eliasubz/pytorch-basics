from pathlib import Path
from time import time

import torch

def compile_shader():
    metal_source = (Path(__file__).parent / "matMulRow.metal").read_text()
    return torch.mps.compile_shader(metal_source)

def matrix_multiplication (lib, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.device == B.device
    assert A.dtype == B.dtype == torch.float32
    assert A.size() == B.size()

    n = A.size()[0]
    print(n)
    C = torch.empty_like(A)
    lib.matrixMulRowKernel(A, B, C, n, threads=n, group_size=256)
    print("hello")
    return C




def main():
    lib = compile_shader()

    DEVICE = "mps"
    DTYPE = torch.float32
    size = 1024

    M = torch.randn((size, size)).to(DEVICE, DTYPE)
    N = torch.randn((size, size), dtype=DTYPE, device=DEVICE)
    
    start = time()
    y_custom_kernel = matrix_multiplication(lib, M, N)
    torch.mps.synchronize()
    stop = time()
    print(f"Metal custom kernel multiply: {stop - start:.4f}s")

    start = time()
    y_torch = M @ N
    torch.mps.synchronize()
    stop = time()
    print(f"PyTorch multiply: {stop - start:.4f}s")

    print("Match:", torch.allclose(y_custom_kernel, y_torch))
    print("Y:", y_custom_kernel[:10])


if __name__ == "__main__":
    main()
