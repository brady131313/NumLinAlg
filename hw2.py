import os
import time
import argparse

import util
import matrix
import solvers

def hw2(fileName, maxIter, tolerance, iterMatrix, display, displayResidual):
    start = time.time()
    with open(util.getMatrixFile(fileName)) as file:
        A = matrix.Sparse.fromFile(file)
    end = time.time()

    print(f"Time to read matrix from file was {end - start} seconds")

    #Generate random solution vector
    x = matrix.Vector.fromRandom(A.columns, 0, 5)
    b = A.multVec(x)

    #Generate first iteration
    xInit = matrix.Vector(A.columns)

    #Get proper iteration matrix
    if iterMatrix.lower() == "l1":
        iterMatrix = solvers.IterMatrix.l1Smoother
    elif iterMatrix.lower() == "fgs":
        iterMatrix = solvers.IterMatrix.forwardGaussSeidel
    elif iterMatrix.lower() == "bgs":
        iterMatrix = solvers.IterMatrix.backwardGaussSeidel
    elif iterMatrix.lower() == "sgs":
        iterMatrix = solvers.IterMatrix.symmetricGaussSeidel
    else:
        raise Exception("No valid iteration matrix selected")

    #Solve system using stationary iterative method
    start = time.time()
    xResult, iterations, residual = solvers.stationaryIterative(A, b, xInit, maxIter, tolerance, iterMatrix, displayResidual)
    end = time.time()

    if display:
        util.compareVectors(x, xResult)

    print(f"Iterations = {iterations}, Residual = {residual} on exit")
    print(f"Time to iteratively solve system was {end - start} seconds using {iterMatrix.name}")



parser = argparse.ArgumentParser()
parser.add_argument("filename", help="matrix to be factored and solved", type=str)
parser.add_argument("-i", dest="maxIter", default=1000, action='store', type=int, help="Max number of iterations")
parser.add_argument("-t", dest="tolerance", default=1e-6, action='store', type=float, help="Tolerance needed for convergence")
parser.add_argument("-B", dest="iterMatrix", default="l1", action='store', type=str, help="Type of iteration matrix to use")
parser.add_argument("-d", dest="display", action='store_true', help="display result")
parser.add_argument("-r", dest="residual", action='store_true', help="Display residual during each iteration")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("Matrix filename must be supplied")
else:
    hw2(args.filename, args.maxIter, args.tolerance, args.iterMatrix, args.display, args.residual)

