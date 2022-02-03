import numpy as np
from numba import cuda
import timeit

@cuda.jit
def add(n, x, y):
    i = cuda.grid(1)
    if i < x.shape[0]:
        y[i] = x[i] + y[i]

RUNS = 100
N = 1<<20

# initialize the data on the host
x = np.ones((N,))
y = 2*np.ones((N,))

# prefetch the date to the GPU
d_x = cuda.to_device(x)
d_y = cuda.to_device(y)


# grid dimensions
threads_per_block = 1024
blocks = (N + threads_per_block - 1) // threads_per_block

# Precompile
add[blocks, threads_per_block](N, d_x, d_y)

start = timeit.default_timer()
for _ in range(RUNS):
    add[blocks, threads_per_block](N, d_x, d_y)
end = timeit.default_timer()
print('Time per run', (end-start)/RUNS)

# fetch data from GPU
d_y.copy_to_host(y)

# check for errors
maxError = np.max(np.abs(y-2-RUNS-1))
print('Max error:', maxError)
