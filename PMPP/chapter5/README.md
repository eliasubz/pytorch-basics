# PMPP Chapter 5 — Tiled Matrix Multiplication with Phases

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliasubz/pytorch-basics/blob/main/PMPP/chapter5/tiled_matmul.ipynb)

> Set runtime to **T4 GPU**: Runtime → Change runtime type → T4 GPU

## What's in the notebook

| Cell | Topic |
|------|-------|
| Naive kernel | Baseline — all reads from global DRAM, CGMA = 1.0 |
| Tiled kernel (annotated) | Phase loop with `__shared__`, two `__syncthreads()` barriers |
| Benchmark | `cudaEvent` timing — naive vs tiled on 512×512 |
| Ex 5.6.3 | Why both sync barriers are required |

## Key concept

```
CGMA ratio = TILE_WIDTH
```

With a 16×16 tile, each float from global memory is reused 16 times inside shared memory → **16× fewer DRAM accesses** vs the naive kernel.
