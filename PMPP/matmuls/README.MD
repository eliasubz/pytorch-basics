# Matmul

This work is intended to go through each optimization from pmpp and implementing it in cuda, triton and CuTe DSL. 

#### Ch 3 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliasubz/pytorch-basics/blob/main/PMPP/matmuls/ch3.ipynb)

Giga nested loop inside of **one** thread — no parallelism at all. Then the two outer loops become grid dimensions so each thread computes one output element of P and only the inner loop stays. Both implemented in CUDA and Triton + benchmarking.

#### Ch 4
The only change this chapter introduced are the correct block,thread accesses becaues the ch3 kernel could only use one block, which limits the size of the matrices that can be multiplied.

#### Ch 5
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliasubz/pytorch-basics/blob/main/PMPP/matmuls/ch5.ipynb)
