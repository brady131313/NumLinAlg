import os
import time

import matrix
import decomp

def getMatrixFile(name):
    scriptDir = os.path.dirname(__file__)
    relDir = "samples/" + name
    filePath = os.path.join(scriptDir, relDir)
    return filePath



def main():
    #with open(getMatrixFile("jgl009.mtx")) as file:
    #    A = matrix.Dense.fromFile(file)

    A = matrix.Dense(3, 3, [[1, 2, -1],
                            [2, 1, 1],
                            [1, 2, 1]])

    start = time.time()

    Q, R = decomp.QRDecompositon(A)

    end = time.time()
    print(A)
    print(f"Total operation time: {end - start}")



main()
