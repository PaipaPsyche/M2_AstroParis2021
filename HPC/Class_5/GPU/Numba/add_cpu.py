import numpy as np
from numba import njit,prange
import timeit

@njit(parallel=True)
def add(n, x, y):
    for i in prange(n):
        y[i] = x[i] + y[i]

RUNS = 500
N = 1<<20

x = np.ones((N,))
y = 2*np.ones((N,))

start = timeit.default_timer()
for _ in range(RUNS):
    add(N, x, y)
end = timeit.default_timer()
print('Time per run', (end-start)/RUNS)

maxError = np.max(np.abs(y-2-RUNS))
print('Max error:', maxError)
