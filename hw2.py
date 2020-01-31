import os
import time
import argparse

import util
import matrix
import solvers

def hw2(fileName, display):
    start = time.time()
    with open(util.getMatrixFile(fileName)) as file:
        A = matrix.Sparse.fromFile(file)
    end = time.time()

    print(f"Time to read matrix from file was {end - start} seconds")

    x = matrix.Vector.fromRandom(A.columns, 0, 5)
    b = A.multVec(x)

    start = time.time()
    xInit = matrix.Vector(A.columns)
    xResult = solvers.stationaryIterative(A, b, xInit, 10000, 0.000001)
    end = time.time()

    print(f"Time to iteratively solve system was {end - start} seconds")

    if display:
        util.compareVectors(x, xResult)


'''
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="matrix to be factored and solved", type=str)
parser.add_argument("-d", dest="display", action='store_true', help="display result")
args = parser.parse_args()

if true or not args.filename or len(args.filename) == 0:
    print("Matrix filename must be supplied")
else:
    hw2(args.filename, args.display)
'''
hw2("25.mtx", True)