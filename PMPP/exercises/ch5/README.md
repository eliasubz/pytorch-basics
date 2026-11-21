# Chapter 5 (Metal)

## Code

In this chapter we implement the tiled matrix multiplication algorithm in Metal. [matMul.metal](matMul.metal) contains a naive kernel and a tiled kernel (`TILE_WIDTH = 16`, with boundary checks so any width works); [matMulBenchmark.py](matMulBenchmark.py) compiles them with `torch.mps.compile_shader`, benchmarks both on 2048 × 2048 matrices and checks them against `M @ N`.

```bash
python matMulBenchmark.py
```

On an Apple M5:

```
Average time for matMulTiledKernel: 26.3954 ms
Average time for matMulKernel: 47.4217 ms
Outputs are approximately the same
```

### CUDA → Metal memory terminology

| CUDA | Metal |
|---|---|
| global memory | `device` address space |
| constant memory | `constant` address space |
| shared memory (`__shared__`) | threadgroup memory (`threadgroup`) |
| local variable / registers | `thread` address space (the default) |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |

## Exercises

### Exercise 1

**Consider matrix addition. Can one use threadgroup memory to reduce the device memory bandwidth consumption? Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.**

No. In matrix addition there is no reuse of values between threads: thread `(0, 0)` reads `M[0, 0]` and `N[0, 0]`, thread `(124, 12)` reads `M[124, 12]` and `N[124, 12]`, and so on. Every element is read exactly once, by exactly one thread. Staging it in threadgroup memory would still need that one device-memory read, plus an extra write to and read from threadgroup memory, so it can only make things slower.

### Exercise 3

**What type of incorrect execution behavior can happen if one forgot to use one or both `threadgroup_barrier()` calls in the tiled kernel ([matMul.metal](matMul.metal), PMPP Fig. 5.9)?**

- **Without the first barrier** (after loading the tile), a thread can start the inner product loop before the other threads have written their elements of `Mds` and `Nds`. It reads stale values from the previous phase or uninitialized memory, so `Pvalue` is wrong. This is a *read-after-write* hazard.
- **Without the second barrier** (after the inner product loop), a fast thread can move on to the next phase and overwrite its element of `Mds`/`Nds` while slower threads are still reading the current tile. This is a *write-after-read* hazard.

These bugs are timing-dependent, so the kernel could theoretically produce correct results on some runs, sizes although that will probably not happen.

### Exercise 4

**Assuming that capacity is not an issue for registers or threadgroup memory, give one important reason why it would be valuable to use threadgroup memory instead of registers to hold values fetched from device memory? Explain your answer.**

Threadgroup memory is visible to *every* thread in the threadgroup, while registers are private to one thread. A value loaded once from device memory into threadgroup memory can be used by all threads in the threadgroup. In tiled matmul each element of `Mds` is used by `TILE_WIDTH` different threads. If the value lived in one thread's registers, every other thread that needed it would have to load it from device memory again.

### Exercise 5

**For our tiled matrix-matrix multiplication kernel, if we use a 32 × 32 tile, what is the reduction of memory bandwidth usage for input matrices M and N?**

A **32×** reduction.

Without tiling, every thread loads its whole row of `M` and column of `N` from device memory, so each input element is loaded `width` times. With tiling, each element is loaded into threadgroup memory once per threadgroup that needs it and then reused by the 32 threads in that tile row (for `M`) or tile column (for `N`). Each element is therefore loaded `width / 32` times, which is a factor of 32 fewer loads.

(On Apple GPUs a 32 × 32 tile is a threadgroup of 1024 threads, which is exactly the `maxTotalThreadsPerThreadgroup` limit; two `float` tiles use `2 x 32 x 32 x 4 = 8 KB` of the 32 KB threadgroup memory.)

### Exercise 6

**Assume that a Metal kernel is dispatched with 1000 threadgroups, each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?**

Local variables live in the `thread` address space, so there is one per thread: `1000 x 512 = 512,000` versions.

### Exercise 7

**In the previous question, if a variable is declared as a `threadgroup` variable, how many versions of the variable will be created through the lifetime of the execution of the kernel?**

One per threadgroup: `1000` versions.

### Exercise 8

**Consider performing a matrix multiplication of two input matrices with dimensions N × N. How many times is each element in the input matrices requested from device memory when:**

**a. There is no tiling?**

`N` times. Element `M[r][k]` is needed by every output element in row `r` of `P`, and there are `N` of them. Each of those threads loads it separately. The same holds for every element of `N`, which is needed by all `N` outputs in its column.

**b. Tiles of size T × T are used?**

`N / T` times. The element is loaded into threadgroup memory once per threadgroup that needs it, and then all `T` threads in that tile row or column read it from threadgroup memory. The `N` outputs that need it are spread over `N / T` threadgroups, so loads drop by a factor of `T`.

### Exercise 9

**A kernel performs 36 floating-point operations and seven 32-bit device memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute-bound or memory-bound.**

The kernel's arithmetic intensity is `36 FLOP / (7 x 4 B) = 36 / 28 ≈ 1.29 FLOP/B`. A device is compute-bound for this kernel if it can't do the FLOPs as fast as memory delivers the bytes, i.e. when the intensity is above the device's `peak FLOPS / peak bandwidth` ratio.

**a. Peak FLOPS = 200 GFLOPS, peak memory bandwidth = 100 GB/second**

- Compute limit: `200 x 10^9 / 36 ≈ 5.56 x 10^9` threads/s
- Memory limit: `100 x 10^9 / 28 ≈ 3.57 x 10^9` threads/s

Memory is the lower limit (the device ratio is `200 / 100 = 2 FLOP/B > 1.29`), so the kernel is **memory-bound**.

**b. Peak FLOPS = 300 GFLOPS, peak memory bandwidth = 250 GB/second**

- Compute limit: `300 x 10^9 / 36 ≈ 8.33 x 10^9` threads/s
- Memory limit: `250 x 10^9 / 28 ≈ 8.93 x 10^9` threads/s

Compute is the lower limit (the device ratio is `300 / 250 = 1.2 FLOP/B < 1.29`), so the kernel is **compute-bound**, though only barely.

### Exercise 10

**To manipulate tiles, a new Metal programmer has written a kernel that will transpose each tile in a matrix. The tiles are of size BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of matrix A is known to be a multiple of BLOCK_WIDTH. The dispatch and kernel code are shown below. BLOCK_WIDTH is known at compile time and could be set anywhere from 1 to 20.**

```swift
1  let tg = MTLSize(width: BLOCK_WIDTH, height: BLOCK_WIDTH, depth: 1)
2  let grid = MTLSize(width: A_width / BLOCK_WIDTH, height: A_height / BLOCK_WIDTH, depth: 1)
3  encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
```

```cpp
4  kernel void BlockTranspose(device float* A_elements [[buffer(0)]],
5                             constant int& A_width     [[buffer(1)]],
6                             uint2 tgid [[threadgroup_position_in_grid]],
7                             uint2 tid  [[thread_position_in_threadgroup]]) {
8      threadgroup float blockA[BLOCK_WIDTH][BLOCK_WIDTH];
9
10     int baseIdx = tgid.x * BLOCK_WIDTH + tid.x;
11     baseIdx += (tgid.y * BLOCK_WIDTH + tid.y) * A_width;
12
13     blockA[tid.y][tid.x] = A_elements[baseIdx];
14
15     A_elements[baseIdx] = blockA[tid.x][tid.y];
16 }
```

**a. Out of the possible range of values for BLOCK_WIDTH, for what values will this kernel execute correctly on the device?**

It is only *guaranteed* to be correct for `BLOCK_WIDTH = 1`. Then the threadgroup is a single thread that reads and writes its own element, so there is nobody to race with.

For `BLOCK_WIDTH ≤ 5`, the whole threadgroup (at most 25 threads) fits in one 32-wide SIMD-group, so it will *often* happen to work, because the threads tend to run in lockstep. As discussed in chapter 4, exercise 5, that is not guaranteed. Without a barrier the compiler is free to reorder the threadgroup memory accesses, so it's still a bug. From 6 upwards the threadgroup spans several SIMD-groups and the kernel breaks in practice. (All sizes up to 20 × 20 = 400 threads are valid dispatches on Apple GPUs, so the size limit is not the problem.)

**b. If the code does not execute correctly for all BLOCK_WIDTH values, what is the root cause of this incorrect execution behavior? Suggest a fix to the code to make it work for all BLOCK_WIDTH values.**

Line 15 reads `blockA[tid.x][tid.y]`, an element written by a *different* thread on line 13. Nothing guarantees that thread has done its write yet. This is a read-after-write race on threadgroup memory. The fix is a barrier between the write and the read:

```cpp
13     blockA[tid.y][tid.x] = A_elements[baseIdx];
14     threadgroup_barrier(mem_flags::mem_threadgroup);
15     A_elements[baseIdx] = blockA[tid.x][tid.y];
```

No second barrier is needed: each thread only writes back to its own `baseIdx`, and nobody reads device memory after that.

### Exercise 11

**Consider the following Metal kernel and the corresponding host code that dispatches it:**

```cpp
1  kernel void foo_kernel(device const float* a [[buffer(0)]],
2                         device float* b       [[buffer(1)]],
3                         uint i   [[thread_position_in_grid]],
4                         uint tid [[thread_position_in_threadgroup]],
5                         uint threads [[threads_per_grid]]) {
6      float x[4];
7      threadgroup float y_s;
8      threadgroup float b_s[128];
9      for (uint j = 0; j < 4; ++j) {
10         x[j] = a[j * threads + i];
11     }
12     if (tid == 0) {
13         y_s = 7.4f;
14     }
15     b_s[tid] = b[i];
16     threadgroup_barrier(mem_flags::mem_threadgroup);
17     b[i] = 2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3]
18             + y_s*b_s[tid] + b_s[(tid + 3) % 128];
19 }
```

```swift
20 let N = 1024
21 encoder.dispatchThreadgroups(MTLSize(width: (N + 128 - 1) / 128, height: 1, depth: 1),
22                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
```

**a. How many versions of the variable `i` are there?**

One per thread. There are `(1024 + 128 - 1) / 128 = 8` threadgroups of 128 threads, so `8 x 128 = 1024` threads and `1024` versions.

**b. How many versions of the array `x[]` are there?**

Also one per thread: `1024`.

**c. How many versions of the variable `y_s` are there?**

`y_s` is in threadgroup memory, so there is one per threadgroup: `8`.

**d. How many versions of the array `b_s[]` are there?**

Also one per threadgroup: `8`.

**e. What is the amount of threadgroup memory used per threadgroup (in bytes)?**

`y_s` is one `float` (4 bytes) and `b_s` is 128 floats (`128 x 4 = 512` bytes), so `4 + 512 = 516` bytes. (The compiler may round this up for alignment. In Metal you can check the real value with the pipeline's `staticThreadgroupMemoryLength`.)

**f. What is the floating-point to device memory access ratio of the kernel (in OP/B)?**

Device memory accesses per thread:

- Line 10: 4 loads from `a`
- Line 15: 1 load from `b`
- Line 17: 1 store to `b`

Floating-point operations per thread (lines 17–18), where six terms are summed:

- 5 multiplications: `2.5f*x[0]`, `3.7f*x[1]`, `6.3f*x[2]`, `8.5f*x[3]`, `y_s*b_s[tid]`
- 5 additions to combine the 6 terms

That's 10 FLOPs. Counting only the loads, there are `5 x 4 = 20` bytes, giving `10 / 20 = 0.5 OP/B`. If the store also counts as an access, there are `6 x 4 = 24` bytes, giving `10 / 24 ≈ 0.42 OP/B`.

### Exercise 12

**Consider a GPU with the following hardware limits per core: 2048 threads, 32 threadgroups, 64K (65,536) registers, and 96 KB of threadgroup memory. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.**

*The book says "KB of shared memory/SM" for the kernels, which would make no sense as a kernel property; we read it as per threadgroup.*

**a. The kernel uses 64 threads/threadgroup, 27 registers/thread, and 4 KB of threadgroup memory/threadgroup.**

- Threads: `2048 / 64 = 32` threadgroups, exactly the 32 limit.
- Registers: `2048 x 27 = 55,296` ≤ 65,536, fine.
- Threadgroup memory: `32 x 4 KB = 128 KB` > 96 KB. Only `96 / 4 = 24` threadgroups fit.

`24 x 64 = 1536` threads: **75% occupancy**, limited by **threadgroup memory**.

**b. The kernel uses 256 threads/threadgroup, 31 registers/thread, and 8 KB of threadgroup memory/threadgroup.**

- Threads: `2048 / 256 = 8` threadgroups, under the 32 limit.
- Registers: `2048 x 31 = 63,488` ≤ 65,536, fine.
- Threadgroup memory: `8 x 8 KB = 64 KB` ≤ 96 KB, fine.

`8 x 256 = 2048` threads: **full occupancy (100%)**.
