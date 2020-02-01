import os
import time

import matrix
import decomp
import solvers
import util

def main():
    #with open(getMatrixFile("jgl009.mtx")) as file:
    #    A = matrix.Dense.fromFile(file)

    C = matrix.Dense(4, 4, [[2, 1, 0, 3],
                            [0, 4, 0, 0],
                            [0, 0, 2, 1],
                            [0, 0, 0, 1]])

    A = matrix.Sparse.fromDense(C)

    xOrig = matrix.Vector.fromRandom(C.columns, 0, 5)
    b = A.multVec(xOrig)

    start = time.time()
    x = solvers.backwardSparse(A, b)
    end = time.time()

    util.compareVectors(A.multVec(x), b)

    print(f"Total operation time: {end - start}")



main()
