import argparse
import time

import util
from linalg import stationaryIterative, l1Solver, fgsSolver, bgsSolver, sgsSolver
from matrix import Sparse, Vector


def hw2(fileName, O, maxIter, tolerance, iterMatrix, display, displayResidual):
    with open(util.getMatrixFile(fileName)) as file:
        A = Sparse.fromFile(file, O)

    # Generate random solution vector
    x = Vector.fromRandom(A.columns, 0, 5)
    b = A.multVec(x)

    # Generate first iteration
    xInit = Vector(A.columns)

    # Get proper iteration matrix
    if iterMatrix.lower() == "l1":
        method = "l1 Smoother"
        iterSolver = l1Solver(A)
    elif iterMatrix.lower() == "fgs":
        method = "Forward Gauss Seidel"
        iterSolver = fgsSolver(A)
    elif iterMatrix.lower() == "bgs":
        method = "Backward Gauss Seidel"
        iterSolver = bgsSolver(A)
    elif iterMatrix.lower() == "sgs":
        method = "Symmetric Gauss Seidel"
        iterSolver = sgsSolver(A)
    else:
        raise Exception("No valid iteration matrix selected")

    # Solve system using stationary iterative method
    start = time.time()
    xResult, iterations, residual, accuracy = stationaryIterative(A, b, xInit, maxIter, tolerance, iterSolver,
                                                                  displayResidual)
    end = time.time()

    convergence = "(Convergence)" if maxIter != iterations else ""

    if display:
        util.compareVectors(x, xResult)

    print(f"\nB Matrix   = {method}")
    print(f"Iterations = {iterations} {convergence}")
    print(f"Residual   = {residual}")
    print(f"Accuracy   = {accuracy}")
    print(f"Time to solve was {end - start}\n")


parser = argparse.ArgumentParser()
parser.add_argument("filename", help="matrix to be factored and solved", type=str)
parser.add_argument("-i", dest="maxIter", default=1000, action='store', type=int, help="Max number of iterations")
parser.add_argument("-t", dest="tolerance", default=1e-6, action='store', type=float,
                    help="Tolerance needed for convergence")
parser.add_argument("-B", dest="iterMatrix", default="l1", action='store', type=str,
                    help="Type of iteration matrix to use")
parser.add_argument("-d", dest="display", action='store_true', help="display result")
parser.add_argument("-r", dest="residual", action='store_true', help="Display residual during each iteration")
parser.add_argument("-O", dest="O", default=0, action='store', type=int, help="Offset for file")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("Matrix filename must be supplied")
else:
    hw2(args.filename, args.O, args.maxIter, args.tolerance, args.iterMatrix, args.display, args.residual)
