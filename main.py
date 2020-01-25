import os
import time

import matrix
import decomp
import solvers

def getMatrixFile(name):
    scriptDir = os.path.dirname(__file__)
    relDir = "samples/" + name
    filePath = os.path.join(scriptDir, relDir)
    return filePath



def main():
    #with open(getMatrixFile("jgl009.mtx")) as file:
    #    A = matrix.Dense.fromFile(file)

    C = matrix.Dense(4, 4, [[1, 0, 0, 0],
                            [3, 4, 0, 0],
                            [6, 7, 8, 0],
                            [1, 2, 3, 4]])

    A = matrix.Sparse.fromDense(C)
    b = matrix.Vector(4, [2, 5, 9, 5])

    print(A)

    start = time.time()
    x = solvers.forwardSparse(A, b)
    end = time.time()

    print(x)
    print(A.multVec(x))
    print(f"Total operation time: {end - start}")



main()
