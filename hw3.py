from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import argparse
import time

import util
from graph import Graph, getVertexAggregate, formCoarse, kMeans

def formCoordinateVectors(L, d):
    converted = csr_matrix((L.data, L.colInd, L.rowPtr), (L.rows, L.columns))
    converted = converted.asfptype()
    
    eigs, vecs = eigsh(converted, d, which='SM', tol=1e-3)
    return vecs

def getColors(X, P):
    colors = [0 for _ in range(len(X))]

    for i in range(P.rows):
        for k in range(P.rowPtr[i], P.rowPtr[i + 1]):
            j = P.colInd[k]
            #colors[i] = 1 - (1/(1 + j))
            colors[i] = j

    colorMap = cm.get_cmap('Dark2')
    return colorMap(colors)

def visualizeClusters2d(X, P):
    xCoords = [x[0] for x in X]
    yCoords = [x[1] for x in X]
    colors = getColors(X, P)

    size, alpha = (75, 0.4) if len(xCoords) < 1250 else (50, 0.2)
    
    plt.scatter(xCoords, yCoords, c=colors, s=size, alpha=alpha, marker="o")

    plt.show()

def visualizeClusters3d(X, P):
    xCoords = [x[0] for x in X]
    yCoords = [x[1] for x in X]
    zCoords = [x[2] for x in X]
    colors = getColors(X, P)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    size, alpha = (75, 0.4) if len(xCoords) < 1250 else (50, 0.2)

    ax.scatter(xCoords, yCoords, zCoords, c=colors, s=size, alpha=alpha)

    plt.show()

def hw3(filename, K, d, maxIter, tolerance, p):
    with open(util.getMatrixFile(filename)) as file:
        g = Graph.fromFile(file)

    #if K > g.adjacency.rows or d < K or d > g.adjacency.rows:
    #    raise Exception("k << n, d >= k, d < n must hold")

    L = g.getLaplacian()
    X = formCoordinateVectors(L, d)

    clusters, iterations, delta = kMeans(X, K, d, maxIter, tolerance)
    print(f"{iterations} iterations to find clusters")

    for i in range(len(clusters)):
        print(f"|A{i + 1}| = {len(clusters[i])}")

    start = time.time()
    vertexAggregate = getVertexAggregate(X, clusters)
    end = time.time()
    print(f"Elapsed: {end - start}")

    coarse = formCoarse(vertexAggregate, L)


    if d == 2:
        visualizeClusters2d(X, vertexAggregate)
    elif d == 3:
        visualizeClusters3d(X, vertexAggregate)
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

