#----------------------------------------------------------------
# File:     gauss_solve.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@arizona.edu)
# Date:     Thu Sep 26 10:38:32 2024
# Copying:  (C) Marek Rychlik, 2020. All rights reserved.
# 
#----------------------------------------------------------------
# A Python wrapper module around the C library libgauss.so

import ctypes
import numpy as np

gauss_library_path = './libgauss.so'

def unpack(A):
    """ Extract L and U parts from A, fill with 0's and 1's """
    n = len(A)
    L = [[A[i][j] for j in range(i)] + [1] + [0 for j in range(i+1, n)]
         for i in range(n)]

    U = [[0 for j in range(i)] + [A[i][j] for j in range(i, n)]
         for i in range(n)]

    return L, U

def lu_c(A):
    """ Accepts a list of lists A of floats and
    it returns (L, U) - the LU-decomposition as a tuple.
    """
    # Load the shared library
    lib = ctypes.CDLL(gauss_library_path)

    # Create a 2D array in Python and flatten it
    n = len(A)
    flat_array_2d = [item for row in A for item in row]

    # Convert to a ctypes array
    c_array_2d = (ctypes.c_double * len(flat_array_2d))(*flat_array_2d)

    # Define the function signature
    lib.lu_in_place.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))

    # Modify the array in C (e.g., add 10 to each element)
    lib.lu_in_place(n, c_array_2d)

    # Convert back to a 2D Python list of lists
    modified_array_2d = [
        [c_array_2d[i * n + j] for j in range(n)]
        for i in range(n)
    ]

    # Extract L and U parts from A, fill with 0's and 1's
    return unpack(modified_array_2d)

def lu_python(A):
    n = len(A)
    for k in range(n):
        for i in range(k,n):
            for j in range(k):
                A[k][i] -= A[k][j] * A[j][i]
        for i in range(k+1, n):
            for j in range(k):
                A[i][k] -= A[i][j] * A[j][k]
            A[i][k] /= A[k][k]

    return unpack(A)

class NoImpementationInC(Exception):
    pass

def plu(A,use_c):
  if use_c:
    print('Attempting to use C')
    #raise(NoImpementationInC)
    lib = ctypes.CDLL(gauss_library_path)
    lib.plu.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.POINTER(ctypes.c_int)]
    lib.plu.restype = None
    
    # Create a sample matrix A (3x3) in Python using numpy
    #A = np.array([[4, 9, 10], [14, 30, 34], [2, 3, 3]], dtype=np.float64)
    A = np.array(A)
    n = len(A)
    
    # Create a permutation vector P
    P = np.zeros(n, dtype=np.int32)
    
    # Create a ctypes-compatible matrix A
    A_ctypes = (ctypes.POINTER(ctypes.c_double) * n)(*[
        (ctypes.c_double * n)(*row) for row in A
    ])
    
    # Call the C function
    lib.plu(n, A_ctypes, P.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    # Print the results
    print("Permutation vector P:", P)
    
    print("Lower triangular matrix L:")
    for i in range(n):
        for j in range(n):
            print(f"{A[i][j]:.6f}", end=" ")
        print()
    
    L,U = lu_c(A)
    #L = np.array(L).tolist()
    #U = np.array(U).tolist()
    #print(type(P))
    #print(P)
    #print(type(L))
    #print(L)
    #print(type(U))
    #print(U)
    print('error is in return validation')
    return P.tolist(),L,U

  else:
    #A = np.array([[4,9,10],[14,30,34],[2,3,3]])
    A = np.array(A)
    #print(A)
    n = len(A)
    P = np.arange(n)
    L = np.zeros_like(A).astype(np.float64)
    U = A.astype(np.float64)
    
    for k in range(0,n):
      pivot = np.argmax(U[k:,k])+k
      if k!= pivot:
        print('swapping rows',k,'and',pivot)
        U[[k,pivot]] = U[[pivot,k]]
        P[[k,pivot]] = P[[pivot,k]]
        L[[k, pivot], :k] = L[[pivot, k], :k]
        
      for i in range(k + 1, n):
        multiplier = U[i, k] / U[k, k]
        L[i, k] = multiplier
        U[i, k:n] -= multiplier * U[k, k:n]

    for ii in range(n):
      L[ii,ii] = 1

    print('error is in return validation...')
    return P.tolist(),L.tolist(),U.tolist()
    
def lu(A, use_c=False):
    if use_c:
        return lu_c(A)
    else:
        return lu_python(A)



if __name__ == "__main__":

    def get_A():
        """ Make a test matrix """
        A = [[2.0, 3.0, -1.0],
             [4.0, 1.0, 2.0],
             [-2.0, 7.0, 2.0]]
        return A

    A = get_A()

    L, U = lu(A, use_c = False)
    print(L)
    print(U)

    # Must re-initialize A as it was destroyed
    A = get_A()

    L, U = lu(A, use_c=True)
    print(L)
    print(U)
