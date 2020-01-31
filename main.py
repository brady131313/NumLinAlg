import os
import time

import matrix
import decomp
import solvers

def main():
    #with open(getMatrixFile("jgl009.mtx")) as file:
    #    A = matrix.Dense.fromFile(file)

    C = matrix.Dense(3, 3, [[2, -1, 0],
                            [-1, 2, -1],
                            [0, -1, 2]])

    A = matrix.Sparse.fromDense(C)
    b = matrix.Vector(3, [2, 5, 9])

    print(C)

    start = time.time()
    L, D = decomp.LDLDecomposition(C)
    end = time.time()

    print(L)
    print(D)

    print(f"Total operation time: {end - start}")



main()
