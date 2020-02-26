import argparse

from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from graph.graph import *
from graph import lubys, recursiveLubys, QModularity, modularityMatrix
import util
import plot

def single(filename, modularity, printResult, visualize, d):
    with open(util.getMatrixFile(filename)) as file:
        g = Graph.fromFile(file)

    if modularity:
        w = modularityWeights(g.adjacency, g.edgeVertex)
    else:
        w = g.getRandomWeights()

    P = lubys(g.edgeVertex, w)
    coarse = formCoarse(P, g.adjacency)

    Q = QModularity(g.adjacency, P)
    B = modularityMatrix(g.adjacency)

    if printResult:
        P.visualizeShape()
        print(coarse)

    print(f"{P.columns} clusters, Q = {Q}")

    if visualize:
        L = g.getLaplacian()
        X = plot.formCoordinateVectors(L, d)
        plot.visualize(X, P, d)

def recursive(filename, tau, modularity, printResult, visualize, d):
    with open(util.getMatrixFile(filename)) as file:
        g = Graph.fromFile(file)

    As, Ps = recursiveLubys(g, tau, modularity)

    Pk = formVertexToK1Aggregate(Ps, len(Ps))
    print(f"{Pk.columns} clusters")

    if visualize:
        L = g.getLaplacian()
        X = plot.formCoordinateVectors(L, d)
        plot.visualize(X, Pk, d)



parser = argparse.ArgumentParser()
parser.add_argument("filename", help="graph to be clustered", type=str)
parser.add_argument("-r", dest="r", action='store_true', help="Recursive Lubys")
parser.add_argument("-t", dest="tau", default=8, action='store', type=int, help="Tau shrinkige")
parser.add_argument("-m", dest="m", action='store_true', help="Use modularity matrix weights")
parser.add_argument("-p", dest="p", action='store_true', help="Display matrix")
parser.add_argument("-v", dest="v", action='store_true', help="Visualize clusters")
parser.add_argument("-d", dest="d", default=2, action='store', type=int, help="Dimensions")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("graph must be supplied")
else:
    if args.r:
        recursive(args.filename, args.tau, args.m, args.p, args.v, args.d)
    else:
        single(args.filename, args.m, args.p, args.v, args.d)
