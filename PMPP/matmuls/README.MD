# Matmul

This work is intended to go through each optimization from pmpp and implementing it in cuda, triton and CuTe DSL.

#### 3 Intro to CUDA
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliasubz/pytorch-basics/blob/main/PMPP/matmuls/ch3.ipynb)

Giga nested loop inside of **one** thread and one-block limited thread-per-element kernels. This notebook introduces the most naive implementations of a matmul. The one that in leetcode would timeout during submission because it's so unoptimized.

#### 4 CUDA Threads
The only change this chapter introduced are the correct block,thread accesses becaues the ch3 kernel could only use one block, which limits the size of the matrices that can be multiplied.

#### 5 Memories 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliasubz/pytorch-basics/blob/main/PMPP/matmuls/ch5.ipynb)

This chapter tackles the problem of DRAM vs SRAM. It introduces the idea of __TILING__. Each block loads a chunk of M and N into shared memory so values get reused instead of being re-read from DRAM every time.

#### 6 More on Thread Execution

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliasubz/pytorch-basics/blob/main/PMPP/matmuls/ch6.ipynb)

Warp divergence (sum reduction demo), loop unrolling, data prefetching and thread granularity all stacked on the ch5 tiled kernel. CUDA + Triton + big experiments reproduction of Figure 6.16
