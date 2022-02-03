import numpy as np
from numba import jit
import timeit

#@njit(parallel=True)
# for matrix mutiplication to be possible:
# matrix A of dimensions m x n
# matrix B of dimensions n x p
#def dot_prod( A,m,n, B,p,q,C):
#    for i in prange(m):
#        for j in prange(q):
#            for k in prange(n):
#                C[i,j] = A[i,k] * B[k,j]
#    return C


def dot_prod( A,B):
    C = np.zeros([A.shape[0],B.shape[1]])
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i,j] = A[i,k] * B[k,j]
    return C




@jit
def dot(mat1, mat2):
    s = 0
    mat = np.empty(shape=(mat1.shape[1], mat2.shape[0]), dtype=mat1.dtype)
    for r1 in range(mat1.shape[0]):
        for c2 in range(mat2.shape[1]):
            s = 0
            for j in range(mat2.shape[0]):
                s += mat1[r1,j] * mat2[j,c2]
            mat[r1,c2] = s
    return mat





m,n,p = 6,5,4
matA = np.ones((m,n))
matB = np.ones((n,p)) + 2


start_cpu = timeit.default_timer()
matD = dot_prod(matA,matB)
end_cpu = timeit.default_timer()

print('Time (no numba): ', end_cpu-start_cpu)



start_numba = timeit.default_timer()
#matC = np.zeros([m,p])
#matC = dot_prod(matA,m,n,matB,n,p,matC)
matC = dot(matA,matB)
end_numba = timeit.default_timer()
print('Time (numba): ', end_numba-start_numba)

