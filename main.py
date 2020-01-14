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

    A = matrix.Dense(3, 3, [[3, 1, -1],
                            [0, 3, -1],
                            [0, 0, 1]])
    b = matrix.Vector(3, [3, 2, 1])

    start = time.time()

    #Q, R = decomp.QRDecompositon(A)
    x = solvers.backward(A, b)

    end = time.time()
    print(A)
    print(x)
    print(A.multVec(x))
    print(f"Total operation time: {end - start}")



main()
