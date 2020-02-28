import time
import argparse

import util
from matrix import Sparse, Vector
from linalg import pcg, diagonalPreconditioner, sgsPreconditioner, twolevelPreconditioner, IterMatrix

def hw5(filename, maxIter, tolerance, preconditioner, M, display, displayResidual):
    with open(util.getMatrixFile(filename)) as file:
        A = Sparse.fromFile(file)

    if preconditioner.lower() == "d":
        method = "Diagonal"
        preconditioner = diagonalPreconditioner(A)
    elif preconditioner.lower() == "sgs":
        method = "Symmetric Gauss Seidel"
        preconditioner = sgsPreconditioner(A)
    elif preconditioner.lower() == "2l":
        method = "Symmetric Two Level"
        if M == "l1":
            M = IterMatrix.l1Smoother
            method += " (L1 Smoother)"
        elif M == "fgs":
            M = IterMatrix.forwardGaussSeidel
            method += " (Forward Gauss Seidel)"
        preconditioner = twolevelPreconditioner(A, M)
    else:
        raise Exception("No valid preconditioner selected")

    #Generate random solution vector
    x = Vector.fromRandom(A.columns, 0, 5)
    b = A.multVec(x)

    #Generate first iteration
    xInit = Vector(A.columns)

    start = time.time()
    xResult, iterations, residual = pcg(A, b, xInit, maxIter, tolerance, preconditioner, displayResidual)
    end = time.time()

    if display:
        util.compareVectors(x, xResult)

    print(f"\nIterations = {iterations}, Residual = {residual} on exit")
    print(f"Time to iteratively solve system was {end - start} using {method}\n")


parser = argparse.ArgumentParser()
parser.add_argument("filename", help="Matrix to be solved", type=str)
parser.add_argument("-i", dest="maxIter", default=1000, action='store', type=int, help="Max number of iterations")
parser.add_argument("-t", dest="tolerance", default=1e-6, action='store', type=float, help="Tolerance")
parser.add_argument("-B", dest="preconditioner", default="D", action='store', type=str, help="Preconditioner method to use")
parser.add_argument("-M", dest="M", default="l1", action='store', type=str, help="Convergent method for two level")
parser.add_argument("-d", dest="display", action='store_true', help="Display result")
parser.add_argument("-r", dest="residual", action='store_true', help="Display residual each iteration")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("Matrix filename must be supplied")
else:
    hw5(args.filename, args.maxIter, args.tolerance, args.preconditioner, args.M, args.display, args.residual)