import os
import time
import argparse

import numpy as np
from scipy import linalg

import util
from matrix import Sparse, Dense, Vector
from linalg import forwardSparse, backwardSparse


def hw1(fileName, O, display):
    with open(util.getMatrixFile(fileName)) as file:
        A = Dense.fromFile(file, O)

    # Convert matrix to np array so I can use scipy factorization
    A = np.array(A.data)

    # Get LDLt factorization
    start = time.time()
    L, D, P = linalg.ldl(A)
    U = np.transpose(L)
    end = time.time()

    print(f"\nTime to factor input matrix was {end - start} seconds")

    # Convert to CSR format used in my library
    L = Sparse.fromDense(Dense(L.shape[0], L.shape[1], L.tolist()))
    U = Sparse.fromDense(Dense(U.shape[0], U.shape[1], U.tolist()))

    # Generate random solution vector
    x = Vector.fromRandom(L.columns, 0, 5)

    # Resulting b matrix times x vector
    bL = L.multVec(x)
    bU = U.multVec(x)

    # Solve lower triangular system
    start = time.time()
    r1 = forwardSparse(L, bL)
    end = time.time()

    print(f"Time to solve lower system was {end - start} seconds")

    # Solve upper triangular system
    start = time.time()
    r2 = backwardSparse(U, bU)
    end = time.time()

    print(f"Time to solve upper system was {end - start} seconds\n")
    if display:
        print("Solution to lower triangular")
        util.compareVectors(x, r1)

        print("\nSolution to upper triangular")
        util.compareVectors(x, r2)


parser = argparse.ArgumentParser()
parser.add_argument("filename", help="matrix to be factored and solved", type=str)
parser.add_argument("-d", dest="display", action='store_true', help="display result")
parser.add_argument("-O", dest="O", default=0, action='store', type=int, help="Offset for file")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("Matrix filename must be supplied")
else:
    hw1(args.filename, args.O, args.display)
