import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import timeit

# Cuda Kernel
mod = SourceModule("""
        __global__ void add(int n, double *x, double *y)
        {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            if ( idx < n)
               y[idx] = x[idx] + y[idx];
            }
        """)

add = mod.get_function("add")

RUNS = 100
N = 1<<20

# initialize the data on the host
x = np.ones((N,))
y = 2*np.ones((N,))

# prefetch the date to the GPU
d_x = gpuarray.to_gpu(x)
d_y = gpuarray.to_gpu(y)


# grid dimensions
threads_per_block = 1024
blocks = (N + threads_per_block - 1) // threads_per_block

# run

start = timeit.default_timer()
for _ in range(RUNS):
    add(np.int32(N), d_x, d_y, block=(threads_per_block, 1, 1), grid=(blocks, 1, 1))
end = timeit.default_timer()
print('Time per run', (end-start)/RUNS)

# fetch data from GPU
y = d_y.get()

# check for errors
maxError = np.max(np.abs(y-2-RUNS))
print('Max error:', maxError)


