import os
import time
import argparse

import numpy as np
from scipy import linalg

import util
import matrix
import solvers

def hw1(fileName, display):
    start = time.time()
    with open(util.getMatrixFile(fileName)) as file:
        A = matrix.Dense.fromFile(file)
    end = time.time()

    print(f"Time to read matrix from file was {end - start} seconds")

    #Convert matrix to np array so I can use scipy factorization
    A = np.array(A.data)

    #Get LDLt factorization
    start = time.time()
    L, D, P = linalg.ldl(A)
    U = np.transpose(L)
    end = time.time()
    
    print(f"Time to factor input matrix was {end - start} seconds")

    #Convert to CSR format used in my library
    start = time.time()
    L = matrix.Sparse.fromDense(matrix.Dense(L.shape[0], L.shape[1], L.tolist()))
    U = matrix.Sparse.fromDense(matrix.Dense(U.shape[0], U.shape[1], U.tolist()))
    end = time.time()

    print(f"Time to convert matricies to CSR was {end - start} seconds")

    #Generate random solution vector
    x = matrix.Vector.fromRandom(L.columns, 0, 5)

    #Resulting b matrix times x vector
    bL = L.multVec(x)
    bU = U.multVec(x)

    #Solve lower triangular system
    start = time.time()
    r1 = solvers.forwardSparse(L, bL)
    end = time.time()

    print(f"Time to solve lower system was {end - start} seconds")
    if display:
        util.compareVectors(x, r1)

    #Solve upper triangular system
    start = time.time()
    r2 = solvers.backwardSparse(U, bU)
    end = time.time()

    print(f"Time to solve upper system was {end - start} seconds")
    if display:
        util.compareVectors(x, r2)


parser = argparse.ArgumentParser()
parser.add_argument("filename", help="matrix to be factored and solved", type=str)
parser.add_argument("-d", dest="display", action='store_true', help="display result")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("Matrix filename must be supplied")
else:
    hw1(args.filename, args.display)