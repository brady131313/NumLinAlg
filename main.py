import os
import time

import matrix

def getMatrixFile(name):
    scriptDir = os.path.dirname(__file__)
    relDir = "samples/" + name
    filePath = os.path.join(scriptDir, relDir)
    return filePath



def main():
    with open(getMatrixFile("s3dkq4m2.mtx")) as file:
        A = matrix.Sparse.fromFile(file)

    start = time.time()

    R = A.multMat(A)

    end = time.time()
    #print(R)
    print(f"Total operation time: {end - start}")



main()
