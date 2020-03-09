import time
import argparse

import util
from matrix import Sparse, Vector
from linalg import composite, l1Solver, fgsSolver, bgsSolver, sgsSolver


def hw6(filename, O, display, residual):
    with open(util.getMatrixFile(filename)) as file:
        A = Sparse.fromFile(file, O)

    # Generate random solution vector
    x = Vector.fromRandom(A.columns, 0, 5)
    b = A.multVec(x)

    # Generate first iteration
    xInit = Vector(A.columns)

    # Convergent composite iterative methods
    components = [l1Solver, fgsSolver, bgsSolver, sgsSolver]
    methods = [f.__name__ for f in components]
    components = [f(A) for f in components]

    start = time.time()
    xResult, residual = composite(A, b, xInit, components, residual)
    end = time.time()

    if display:
        print()
        util.compareVectors(x, xResult)

    print(f"\nConvergent methods = {methods}")
    print(f"Residual           = {residual}")
    print(f"Time to solve was {end - start}\n")


parser = argparse.ArgumentParser()
parser.add_argument("filename", help="Matrix to be solved", type=str)
parser.add_argument("-O", dest="offset", default=0,
                    action='store', type=int, help="Offset for matrix file")
parser.add_argument("-d", dest="display",
                    action='store_true', help="Display result")
parser.add_argument("-r", dest="residual", action='store_true',
                    help="Display residual each iteration")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("Matrix filename must be supplied")
else:
    hw6(args.filename, args.offset, args.display, args.residual)
