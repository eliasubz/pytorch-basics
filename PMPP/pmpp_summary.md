# Programming Massively Parallel Processors

## 1 Introduction

### 1.1 GPUs as Parallel Computers

From sequential Multi-threaded CPUs that prioritize complex control logic and large caches to GPUs that optimiye parallel throughput with massie thread counts and highger memory bandwidth. GPUs have become more accessible and more accurate with double precision Floating-point arithmetic (FPA) (Note: this book is from 2010). Until we got CUDA. 

CUDA is a C/C++ library that uses a familiar language instead of clunky GPU apis that were common before. 

### 1.2 Architecture of a Modern GPU

Each GPU has Streaming Multi-processors (SMs) and within each SM there are multiple streaming processors (SPs). The G80 for example has 128 SPs (16 SMs, each with 8 SPs), each with a  multiply–add (MAD) unit and an additional multiply unit. This apparently sums to a 500 gigaflops. 
Each SP has 96 threads and gives us over 12,000 threads, way more than the 2-4 threads per core that intel CPUs had at that time.

Additionally, each GPU had up to 4 gigabytes of graphics double data rate (GDDR) DRAM, also refered to as global memory (lives on the GPU; not to be confused with system memory). That DRAM memory has a memory bandwidth of 86 GB/s and a communication bandwithd with the CPU of 8 GB/s. 

All of these values have changed to now adays but the proportions have likly stayed the same.

### 1.3  Why More Speed or Parallelism?

Simple argument that speed has a lower growth curve than parallism in terms of GFLOPs per GPU vs CPU. One interesting point is that not all compute heavy processes can be ran efficiently in parallel, but since we want to make AI go BRRR, that's not an issue.

### 1.4  Parallel Programming Languages and Models

I skipped this chapter. I don't care about old languages. They have mentioned an interesting 

### 1.5 Overarching Goals

1. Even though computer architecture is necessary, students are supposed to be able to learn the important topics and algorithms from the book without specific knowledge. This book tries to explain these topics on the go. 
2. Complete abstraction is not possible and for really maximizing the performance it is necessary to understand the underlying GPU arch. (Reminds me of flash-attn 4 that only works on hopper and blackwell GPUs).
3. The way this book works is supposed to help programs become more parallizable as time continues (don't ask me how exactly)

### 1.6 Organization of the book

This section is already very trimmed but let me try. 

1. Introduction 
2. History of GPUs movements, GPGPUs, historic developments deepens understanding of current and future ones as well.
3. Simple CUDA progam going through all steps. (1) Isolate data us by parallelized code and transfer it to computing device, (2) developing and launching a parallel kernel function, (3) transferring data back to host processor. 
4. (-7.) CUDA concepts, thread organization, special memories, factors on performance and precision and accuracy. These chapters help to understand general parallel computing concepts. 
8. (-9.) Case Studies
10. Generalizations from problem formulation to algorithm strategies.
11. OpenCL the "new" programming language.
12. Conclusion

## 2 History of GPU Computing

I want to LEARN CUDAAA.
The inrtoduction claims that understanding the history will help programmers understand architectural choices that were made: massivemultithreading,
relatively small cachememories compared to central processing units (CPUs), bandbandwidth-centricmemoryinterfacedesign; and help project the future evolution of GPUs.

### 2.1 EvolutionofGraphicsPipelines

### 2.1.1 TheEraofFixed-FunctionGraphicsPipelines

### 2.1.2 EvolutionofProgrammableReal-TimeGraphics

### 2.1.3 Unified Graphics and Computing Processors

### 2.1.4 GPGPU:An Intermediate Step

### 2.2 GPU Computing

### 2.2.1 Scalable GPUs

### 2.2.2 Recent Developments

### 2.3 Future Trends


## 3 Introduction to CUDA

CUDA programmer use a host (CPU) and a (or multiple) device(s) which are massively parallel processors. 

### 3.1 Data Parallelism
The example is stated that a dot-product is highly parallelizable, because each output depends on one row and column and doesn't change the solution for any other value. If we have two 12.000^2 matrices the G80 could solve it in its over 12.000 threads efficiently. 

### 3.1 Excercise  

If each dot product in a 1000×1000 matrix multiplication is assigned to one CUDA thread, and each thread must read 1000 elements from M and 1000 from N, how many total memory reads occur globally? How does this expose the critical role of shared memory, even though the text doesn't mention it here?

### 3.2 CUDA Program Structure
A CUDA program is a unified source code for both host and device code. The NVIDIA C compiler (nvcc) seperates the two into ANSI C code that is compiled with the hosts standard C compiler and ANSI C code that is extended with keywords for labeling data-parallel functions, called kernels. 

The kernel initializes a big number of threads to exploit parallelism. In the 1000x1000 matrix multiplication example we would initiate 1,000,000 threads. 

These threads take very few cycles to generate in contrast to CPU threads.

### 3.3 A Matrix-Multiplication Example

Here we just see a simple CPU implementation in normal C with three loops. We also get to see a simple scafolding for CUDA programs.

### 3.4 Device Memories and Data Transfer

Allocate Memory on the device transfer it with an API and after execution the memory has to be transfered back and the memory has to be freed up again. 
cudaMalloc() allocates a a piece of memory to the global memory in the device. To copy data we use the cudaMemcpy() function. 

### 3.5 Kernel Function and Threading

Because the same kernel function runs on each thread, CUDA is an instance of the single-program, multiple-data (SPMD) parallel programming style. 
In front of a kernel we use the "\__global__\" keyword to show the host that this method generates a grid of threads on the device.
In order for each kernel in a thread to disinguish itself it calls the threadIdx.x or threadIdx.y variables that give it its coordinates (0,0 or 1,20). They can be used together with the blockDim to access different partitions of the data in global memory.
The two outerloops in the CPU implementation are now actually the threads x and y dimensions in the grid. 

### 3.6 Summary
Contains mostly the same information as the chapter and encourages the reader to consult the CUDA Programming Guide. 


### 3.7 Exercise Solutions 
3.1 It would be 2B reads which explains why fast memory bandwidth is the main accelerator for good GPU performance

## 4 CUDA Threads

This chapter presents more details on th eorganization,resource assignment, and scheduling of threads in a grid.

### 4.1 CUDA Thread Organization