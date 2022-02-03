#!/usr/bin/python

from numba import njit, prange
import numpy as np
import timeit

def poly(P, x):
    res = 0
    power = 1
    for i in range(P.shape[0]):
        res += P[i]*power
        power *= x
    return res

@njit(parallel=True)
def poly_parallel(P, x):
    res = 0
    power = 1
    for i in prange(P.shape[0]):
        res += P[i]*power
        power *= x
    return res

SIZE = 16
P = np.ones((SIZE,))
print(poly(P, 2), 2**SIZE-1)
print(poly_parallel(P, 2), 2**SIZE-1)

#dont_run_the_code_before_predicting_the_result!
