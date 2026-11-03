from pathlib import Path
from time import time

import torch

def compile_shader():
    metal_source = (Path(__file__).parent / "vecMul.metal").read_text()
    return torch.mps.compile_shader(metal_source)

def vector_multiplication (lib, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.device == B.device and A.device.type == 'mps'
    assert A.dtype == B.dtype == torch.float32
    assert A.size() == B.size()

    n = A.size()
    C = torch.empty_like(A)
    lib.vecMulKernel(A, B, C, n, threads=n, group_size=256)
    return C



def main():
    lib = compile_shader()

    DEVICE = "mps"
    DTYPE = torch.float32
    NUM_ELEMENTS = 500_000

    A = torch.arange(NUM_ELEMENTS).to(DEVICE, DTYPE)
    B = torch.arange(NUM_ELEMENTS).to(DEVICE, DTYPE)

    start = time()
    y_custom_kernel = vector_multiplication(lib, A, B)
    torch.mps.synchronize()
    stop = time()
    print(f"Metal custom kernel multiply: {stop - start:.4f}s")

    start = time()
    y_torch = A * B
    torch.mps.synchronize()
    stop = time()
    print(f"PyTorch multiply: {stop - start:.4f}s")

    print("Match:", torch.allclose(y_custom_kernel, y_torch))
    print("Y:", y_custom_kernel[:10])


if __name__ == "__main__":
    main()
