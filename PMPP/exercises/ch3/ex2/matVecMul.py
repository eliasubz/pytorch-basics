from pathlib import Path
from time import time

import torch

def compile_shader():
    metal_source = (Path(__file__).parent / "matVecMul.metal").read_text()
    return torch.mps.compile_shader(metal_source)

def matVecMul (lib, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.device == B.device and A.device.type == 'mps'
    assert A.dtype == B.dtype == torch.float32

    assert A.size(0) == B.size(0)

    n = A.size(0)
    C = torch.empty_like(B)
    lib.matVecMulKernel(A, B, C, n, threads=n, group_size=256)
    return C



def main():
    lib = compile_shader()

    DEVICE = "mps"
    DTYPE = torch.float32
    NUM_ELEMENTS = 5000

    A = torch.randn((NUM_ELEMENTS, NUM_ELEMENTS)).to(DEVICE, DTYPE)
    B = torch.randn((NUM_ELEMENTS, 1)).to(DEVICE, DTYPE)

    start = time()
    y_custom_kernel = matVecMul(lib, A, B)
    torch.mps.synchronize()
    stop = time()
    print(f"Metal custom kernel multiply: {stop - start:.4f}s")

    start = time()
    y_torch = A @ B
    torch.mps.synchronize()
    stop = time()
    print(f"PyTorch multiply: {stop - start:.4f}s")

    print("Match:", torch.allclose(y_custom_kernel, y_torch, rtol=1e-2))
    print("Y_kernel:", y_custom_kernel[-10:])
    print("Y:", y_torch[-10:])


if __name__ == "__main__":
    main()
