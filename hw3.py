import argparse
import time

import util
import plot
from graph import Graph, getVertexAggregate, formCoarse, kMeans, QModularity

def printGraph(g):
    g.adjacency.visualizeShape()
    g.edgeVertex.visualizeShape()
    g.getVertexEdge().visualizeShape()
    print(g.getDegree())
    g.getEdgeEdge().visualizeShape()
    print(g.getLaplacian())

def hw3(filename, O, K, d, maxIter, tolerance, p, G):
    with open(util.getMatrixFile(filename)) as file:
        g = Graph.fromFile(file, O)

    if G: printGraph(g)

    L = g.getLaplacian()
    X = plot.formCoordinateVectors(L, d)

    P, iterations, delta, meta = kMeans(X, K, d, maxIter, tolerance)
    coarse = formCoarse(P, L)

    convergence = "(Convergence)" if iterations != maxIter else ""

    if p:
        P.visualizeShape()
        print(coarse)

    print(f"\nIterations = {iterations} {convergence}")

    for i in range(len(meta)):
        print(f"|A{i + 1}| = {meta[i]}")
    print()

    if d == 2 or d == 3:
        plot.visualize(X, P, d)


parser = argparse.ArgumentParser()
parser.add_argument("filename", help="matrix to be factored and solved", type=str)
parser.add_argument("-d", dest="d", default=2, action='store', type=int, help="Dimensions")
parser.add_argument("-K", dest="K", default=2, action='store', type=int, help="Partitions")
parser.add_argument("-i", dest="maxIter", default=1000, action='store', type=int, help="Max number of iterations")
parser.add_argument("-t", dest="tolerance", default=1e-6, action='store', type=float, help="Tolerance needed for convergence")
parser.add_argument("-p", dest="p", action='store_true', help="Display matricies")
parser.add_argument("-G", dest="G", action='store_true', help="Display graphs")
parser.add_argument("-O", dest="O", default=0, action='store', type=int, help="Offset for file")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("Graph filename must be supplied")
else:
    hw3(args.filename, args.O, args.K, args.d, args.maxIter, args.tolerance, args.p, args.G)

