import numpy as np
import cupy as cp
import timeit

SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]
RUNS = 20

numpy_result = {}
cupy_result = {}

# preload functions
A = cp.random.rand(10, 10)
A = A + A
A = cp.dot(A, A)

for size in SIZES:
    A = np.random.rand(size,size)
    B = np.random.rand(size,size)
    
    C = A
    start = timeit.default_timer()
    for _ in range(RUNS):
        C = C + B
    end = timeit.default_timer()
    numpy_result[f'Add_{size}'] = (end-start)/RUNS

    C = A
    start = timeit.default_timer()
    for _ in range(RUNS):
        C = np.dot(C, B)
    end = timeit.default_timer()
    numpy_result[f'Dot_{size}'] = (end-start)/RUNS
    
    A = cp.random.rand(size,size)
    B = cp.random.rand(size,size)

    C = A
    start = timeit.default_timer()
    for _ in range(RUNS):
        C = C + B
    end = timeit.default_timer()
    cupy_result[f'Add_{size}'] = (end-start)/RUNS

    C = A
    start = timeit.default_timer()
    for _ in range(RUNS):
        C = cp.dot(C, B)
    end = timeit.default_timer()
    cupy_result[f'Dot_{size}'] = (end-start)/RUNS
    
print('%(test)-10s|%(numpy)-10s|%(cupy)-10s|%(speedup)-10s|'%{'test':'TEST',
                                                              'numpy':'NumPy',
                                                              'cupy':'CuPy',
                                                              'speedup':'Seedup'
                                                              })
for key in numpy_result:
    print('%(test)-10s|%(numpy)10.4f|%(cupy)10.4f|%(speedup)10.4f|'%{'test': key,
                                                  'numpy': numpy_result[key],
                                                  'cupy': cupy_result[key],
                                                  'speedup': numpy_result[key]/cupy_result[key]
                                                  })


