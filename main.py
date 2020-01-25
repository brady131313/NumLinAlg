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

    C = matrix.Dense(3, 3, [[1, 2, 3],
                            [0, 5, 6],
                            [0, 0, 8]])

    A = matrix.Sparse.fromDense(C)
    b = matrix.Vector(3, [4, 7, 9])

    print(A)

    start = time.time()
    x = solvers.backwardSparse(A, b)
    end = time.time()

    print(x)
    print(A.multVec(x))
    print(f"Total operation time: {end - start}")



main()
