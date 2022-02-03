import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import timeit

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
    d_y = d_y + d_x
end = timeit.default_timer()
print('Time per run', (end-start)/RUNS)

# fetch data from GPU
y = d_y.get()

# check for errors
maxError = np.max(np.abs(y-2-RUNS))
print('Max error:', maxError)


