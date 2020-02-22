import random
import numpy as np

from matrix import Vector
from graph import *

def recursiveLubys(g, tau):
    A = g.adjacency

    As = [A]
    Ps = []

    while True:
        E = fromAdjacencyToEdge(As[-1])
        
        data = [random.random() for _ in range(E.rows)]
        w = Vector(E.rows, data)

        clusters = lubys(E, w)

        Ps.append(getVertexAggregate2(E.columns, clusters))
        As.append(formCoarse(Ps[-1], As[-1]))
        print(As[0].rows, As[-1].rows)

        if As[0].rows >= tau * As[-1].rows or As[-1].rows == 1:
            break

    return [As, Ps]

def lubys(E, w):
    if E.rows != w.dim:
        raise Exception(f"Dimension mismatch: {E.rows}x{E.columns} * {w.dim}x1")

    edgeEdge = E.multMat(E.transpose())

    clusters = []
    accounted = set()

    for i in range(E.rows):
        edge = E.colInd[E.rowPtr[i]:E.rowPtr[i + 1]]
        weight = w.data[i]

        edgeNeighbors = _findEdgeNeighbors(i, edgeEdge)

        if (_isLargestEdge(i, edgeNeighbors, w)):
            clusters.append(edge)
            accounted.add(edge[0])
            accounted.add(edge[1])

    for v in range(E.columns):
        if v not in accounted:
            clusters.append([v])

    return clusters

def _isLargestEdge(edge, neighbors, weights):
    for e in neighbors:
        if weights.data[e] > weights.data[edge]:
            return False

    return True

def _findEdgeNeighbors(e, edgeEdge):
    neighbors = []
    for k in range(edgeEdge.rowPtr[e], edgeEdge.rowPtr[e + 1]):
        j = edgeEdge.colInd[k]
        if e != j:
            neighbors.append(j)

    return neighbors
        

def kMeans(X, K, d, maxIter, tolerance):
    centers = random.choices(X, k=K)
    delta = 0
    delatPrev = 0
    
    for i in range(maxIter):
        aggregates = [[] for _ in range(K)]
        
        for x in X:
            closest = _findClosestCenter(x, centers)
            aggregates[closest].append(x)

        centers = _updateCenters(aggregates, d)

        delatPrev = delta
        delta = _computeDelta(X, aggregates, centers)

        if i > 1 and abs(delta - delatPrev) <= tolerance * delatPrev:
            print("Convergence")
            return (aggregates, i, delta)
    
    print("Max iter")
    return (aggregates, maxIter, delta)


def _computeDelta(X, aggregates, centers):
    delta = 0

    for r in range(len(centers)):
        for i in range(len(aggregates[r])):
            delta += (np.linalg.norm(np.subtract(aggregates[r][i], centers[r]))) ** 2

    return delta
        

def _updateCenters(aggregates, d):
    centers = [None for _ in range(len(aggregates))]

    for r in range(len(aggregates)):
        newCenter = np.array([0] * d)
        
        for x in aggregates[r]:
            newCenter = np.add(newCenter, x)
        
        if len(aggregates[r]) == 0:
            raise Exception("Empty Cluster")

        newCenter = newCenter * (1 / len(aggregates[r]))
        centers[r] = newCenter
    
    return centers


def _findClosestCenter(x, centers):
    closest = (np.linalg.norm(np.subtract(x, centers[0])), 0)

    for i in range(1, len(centers)):
        update = (np.linalg.norm(np.subtract(x, centers[i])), i)
        if update[0] < closest[0]:
            closest = update
        
    return closest[1]

        