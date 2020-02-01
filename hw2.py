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
    xResult, iterations, residual = solvers.stationaryIterative(A, b, xInit, 1000, 1e-6)
    end = time.time()

    if display:
        util.compareVectors(x, xResult)

    print(f"Iterations = {iterations}, Residual = {residual} on exit")
    print(f"Time to iteratively solve system was {end - start} seconds")



parser = argparse.ArgumentParser()
parser.add_argument("filename", help="matrix to be factored and solved", type=str)
parser.add_argument("-d", dest="display", action='store_true', help="display result")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("Matrix filename must be supplied")
else:
    hw2(args.filename, args.display)

