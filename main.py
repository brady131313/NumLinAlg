import os
import time

import matrix
import decomp
import solvers
import util

def main():
    with open(util.getMatrixFile("50.mtx")) as file:
        A = matrix.Sparse.fromFile(file)

    L, D, U = decomp.symmetricGaussSeidel(A)
    L.visualizeShape()
    U.visualizeShape()

    start = time.time()
    end = time.time()


    print(f"Total operation time: {end - start}")



main()
