import os
import time
import argparse

import util
import matrix
import solvers

def hw2(fileName, maxIter, tolerance, display, displayResidual):
    start = time.time()
    with open(util.getMatrixFile(fileName)) as file:
        A = matrix.Sparse.fromFile(file)
    end = time.time()

    print(f"Time to read matrix from file was {end - start} seconds")

    x = matrix.Vector.fromRandom(A.columns, 0, 5)
    b = A.multVec(x)

    start = time.time()
    xInit = matrix.Vector(A.columns)
    xResult, iterations, residual = solvers.stationaryIterative(A, b, xInit, maxIter, tolerance, displayResidual)
    end = time.time()

    if display:
        util.compareVectors(x, xResult)

    print(f"Iterations = {iterations}, Residual = {residual} on exit")
    print(f"Time to iteratively solve system was {end - start} seconds")



parser = argparse.ArgumentParser()
parser.add_argument("filename", help="matrix to be factored and solved", type=str)
parser.add_argument("-i", dest="maxIter", default=1000, action='store', type=int, help="Max number of iterations")
parser.add_argument("-t", dest="tolerance", default=1e-6, action='store', type=float, help="Tolerance needed for convergence")
parser.add_argument("-d", dest="display", action='store_true', help="display result")
parser.add_argument("-r", dest="residual", action='store_true', help="Display residual during each iteration")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("Matrix filename must be supplied")
else:
    hw2(args.filename, args.maxIter, args.tolerance, args.display, args.residual)

