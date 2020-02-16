from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import argparse

import util
import graph
import decomp
import matrix

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
            colors[i] = 1 - (1/(1 + j))

    colorMap = cm.get_cmap('plasma', 12)
    return colorMap(colors)

def visualizeClusters2d(X, P):
    xCoords = [x[0] for x in X]
    yCoords = [x[1] for x in X]
    colors = getColors(X, P)

    plt.scatter(xCoords, yCoords, c=colors, alpha=0.4)

    plt.show()

def visualizeClusters3d(X, P):
    xCoords = [x[0] for x in X]
    yCoords = [x[1] for x in X]
    zCoords = [x[2] for x in X]
    colors = getColors(X, P)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xCoords, yCoords, zCoords, c=colors)

    plt.show()

def hw3(filename, K, d, maxIter, tolerance):
    with open(util.getMatrixFile(filename)) as file:
        g = graph.Graph.fromFile(file)

    if K > g.adjacency.rows or d < K or d > g.adjacency.rows:
        raise Exception("k << n, d >= k, d < n must hold")

    L = g.getLaplacian()
    X = formCoordinateVectors(L, d)

    clusters = decomp.kMeans(X, K, maxIter, tolerance)

    XConv = [matrix.Vector(len(x), list(x)) for x in X]
    for i in range(len(clusters)):
        clusters[i] = [matrix.Vector(len(c), list(c)) for c in clusters[i]]

    vertexAggregate = graph.getVertexAggregate(XConv, clusters)
    
    A = g.getVertexEdge().multMat(g.edgeVertex)
    coarse = graph.formCoarse(vertexAggregate, A)
    
    if d == 2:
        visualizeClusters2d(X, vertexAggregate)
    elif d == 3:
        visualizeClusters3d(X, vertexAggregate)
    else:
        print("Can't display 4d :(")    

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="matrix to be factored and solved", type=str)
parser.add_argument("-d", dest="d", default=2, action='store', type=int, help="Dimensions")
parser.add_argument("-K", dest="K", default=2, action='store', type=int, help="Dimensions")
parser.add_argument("-i", dest="maxIter", default=1000, action='store', type=int, help="Max number of iterations")
parser.add_argument("-t", dest="tolerance", default=1e-3, action='store', type=float, help="Tolerance needed for convergence")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("Grahp filename must be supplied")
else:
    hw3(args.filename, args.K, args.d, args.maxIter, args.tolerance)

