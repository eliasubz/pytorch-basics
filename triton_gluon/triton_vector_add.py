import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr,  
               y_ptr,  
               output_ptr,  # to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    # not sure if this works because we also use the mask already but should not cause problems
    end = min(n_elements, start + BLOCK_SIZE)

    offset = tl.arange(start, end)

    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    
    output = x + y
    tl.store(output_ptr, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']))
    n_elements = output.numel()
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')