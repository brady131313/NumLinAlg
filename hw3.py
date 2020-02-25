from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import numpy as np
import argparse
import time

import util
import plot
from graph import Graph, getVertexAggregate, formCoarse, kMeans

def hw3(filename, K, d, maxIter, tolerance, p):
    with open(util.getMatrixFile(filename)) as file:
        g = Graph.fromFile(file)

    #if K > g.adjacency.rows or d < K or d > g.adjacency.rows:
    #    raise Exception("k << n, d >= k, d < n must hold")

    L = g.getLaplacian()
    X = plot.formCoordinateVectors(L, d)

    clusters, iterations, delta = kMeans(X, K, d, maxIter, tolerance)
    print(f"{iterations} iterations to find clusters")

    for i in range(len(clusters)):
        print(f"|A{i + 1}| = {len(clusters[i])}")

    start = time.time()
    vertexAggregate = getVertexAggregate(X, clusters)
    end = time.time()
    print(f"Elapsed: {end - start}")

    coarse = formCoarse(vertexAggregate, L)

    if d == 2 or d == 3:
        plot.visualize(X, vertexAggregate, d)
    else:
        print("Can't display 4d :(")

    if p:
        vertexAggregate.visualizeShape()
        print(coarse)

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="matrix to be factored and solved", type=str)
parser.add_argument("-d", dest="d", default=2, action='store', type=int, help="Dimensions")
parser.add_argument("-K", dest="K", default=2, action='store', type=int, help="Partitions")
parser.add_argument("-i", dest="maxIter", default=1000, action='store', type=int, help="Max number of iterations")
parser.add_argument("-t", dest="tolerance", default=1e-6, action='store', type=float, help="Tolerance needed for convergence")
parser.add_argument("-p", dest="p", action='store_true', help="Display matricies")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("Grahp filename must be supplied")
else:
    hw3(args.filename, args.K, args.d, args.maxIter, args.tolerance, args.p)

