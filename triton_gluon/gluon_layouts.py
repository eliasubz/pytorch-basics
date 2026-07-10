# Layouts specify how elems in a tensor are distributed
# among the threads in a threadblock, warps -> lanes -> registers

# #elems is a power of 2

from triton.experimental.gluon import language as gl

# Seems like make_block_ptr from triton, ... maybe only because of order!
# Plus in PMPP I have not seen anything about CTAs; whats that.
gl.BlockedLayout(
    size_per_thread=[2, 4], # each thread has a contiguous 2x4 subtile of the tensor
    # stored as registers. 
    threads_per_warp=[16, 2],
    warps_per_cta=[2, 2],
    # Indicates rows first, i.e. row-major
    order=[1, 0],
)
