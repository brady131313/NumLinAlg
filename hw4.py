import argparse

from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from graph.graph import *
from graph import lubys, recursiveLubys
import util

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
            colors[i] = j
    
    #colorMap = cm.get_cmap('Dark2')
    return colors

def visualize(X, P, d):
    if d > 3: return

    xCoords = [x[0] for x in X]
    yCoords = [x[1] for x in X]
    if d == 3:
        zCoords = [x[2] for x in X]

    colors = getColors(X, P)

    size, alpha = (75, 0.4) if len(xCoords) < 1250 else (50, 0.2)

    if d == 2:
        plt.scatter(xCoords, yCoords, c=colors, s=size, alpha=alpha, marker="o")
        plt.show()
    elif d == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xCoords, yCoords, zCoords, c=colors, s=size, alpha=alpha)
        plt.show()

def hw4(filename, r, tau, p, v):
    with open(util.getMatrixFile(filename)) as file:
        g = Graph.fromFile(file)

    w = g.getRandomWeights()

    print("-- SINGLE --")
    clusters = lubys(g.edgeVertex, w)

    P = getVertexAggregate2(g.adjacency.rows, clusters)
    coarse = formCoarse(P, g.adjacency)

    efficiency = (g.adjacency.rows / 2) / len(clusters)

    if p:
        P.visualizeShape()
        print(coarse)
    print(f"{len(clusters)} clusters, {efficiency} efficiency")

    if v:
        L = g.getLaplacian()
        X = formCoordinateVectors(L, 3)
        visualize(X, P, 3)

    if not r: return

    print("\n-- RECURSIVE --")
    
    As, Ps = recursiveLubys(g, tau)
    print(len(As), len(Ps))

    k = len(Ps)
    Pk = formVertexToK1Aggregate(Ps, k)

    (Pk.transpose().multMat(As[0]).multMat(Pk)).visualizeShape()
    As[k].visualizeShape()



parser = argparse.ArgumentParser()
parser.add_argument("filename", help="graph to be clustered", type=str)
parser.add_argument("-r", dest="r", action='store_true', help="Recursive Lubys")
parser.add_argument("-t", dest="tau", default=8, action='store', type=int, help="Tau shrinkige")
parser.add_argument("-p", dest="p", action='store_true', help="Display matrix")
parser.add_argument("-v", dest="v", action='store_true', help="Visualize clusters")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("graph must be supplied")
else:
    hw4(args.filename, args.r, args.tau, args.p, args.v)