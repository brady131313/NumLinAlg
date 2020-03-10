import argparse
import time

import util
from linalg import composite, l1Solver, fgsSolver, bgsSolver, sgsSolver, constructW
from matrix import Sparse, Vector
from graph import Graph, kMeans
import plot


def hw6(filename, O, K, m, maxIter, tolerance, p):
    with open(util.getMatrixFile(filename)) as file:
        G = Graph.fromFile(file, O)

    L = G.getLaplacian()
    W = constructW(L, m)
    
    P, iterations, delta, meta = kMeans(W, K, m, maxIter, tolerance)
    
    if m == 2 or m == 3:
        plot.visualize(W, P, 2)
    


parser = argparse.ArgumentParser()
parser.add_argument("filename", help="matrix to be factored and solved", type=str)
parser.add_argument("-m", dest="m", default=2, action='store', type=int, help="Dimensions")
parser.add_argument("-K", dest="K", default=2, action='store', type=int, help="Partitions")
parser.add_argument("-i", dest="maxIter", default=1000, action='store', type=int, help="Max number of iterations")
parser.add_argument("-t", dest="tolerance", default=1e-6, action='store', type=float, help="Tolerance needed for convergence")
parser.add_argument("-p", dest="p", action='store_true', help="Display matricies")
parser.add_argument("-O", dest="O", default=0, action='store', type=int, help="Offset for file")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("Matrix filename must be supplied")
else:
    hw6(args.filename, args.O, args.K, args.m, args.maxIter, args.tolerance, args.p)
