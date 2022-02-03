from numba import cuda, njit, prange
import numpy as np


@njit(parallel=True)
def plus_one(A):
    """ Adds one to all componenents of A """
    for i in prange(A.shape[0]):
        A[i] += 1

@cuda.jit
def plus_one_cuda(A):
    """ Adds one to all componenents of A """
    i = cuda.grid(1)
    if i < A.shape[0]:
        A[i] += 1 

def numpy_matmul(A, B, C, EXP):
    C = np.dot(A**EXP, B**EXP)

@njit(parallel=True)
def numba_matmul(A, B, C, EXP):
    """Perform square matrix multiplication of C = A * B
    """
    for i in prange(A.shape[0]):
        for j in prange(B.shape[1]):
            tmp = 0.
            for k in prange(A.shape[1]):
                tmp += A[i, k]**EXP * B[k, j]**EXP
            C[i, j] = tmp

SIZE = 256

A = np.random.rand(SIZE)

plus_one(A)
plus_one_cuda[256, 1](A)
