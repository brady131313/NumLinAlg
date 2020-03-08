import random
import numpy as np
import time

from matrix import Vector, Dense
from graph import *

def QModularity(A, P):
    degree = [0 for _ in range(A.rows)]

    for i in range(len(degree)):
        degree[i] = sum(A.data[A.rowPtr[i]:A.rowPtr[i + 1]])

    T = sum(degree)
    Pt = P.transpose()

    Q = 0
    for aggregate in range(Pt.rows):
        vertcies = Pt.colInd[Pt.rowPtr[aggregate]:Pt.rowPtr[aggregate + 1]]
        
        for i in vertcies:
            for j in vertcies:
                q = A.getValue(i, j) - ((degree[i] * degree[j]) / T)
                #print(q, A.getValue(i, j))
                Q += q

    Q /= T
    
    return Q

def modularityMatrix(A):
    degree = Dense(A.rows, A.rows)
    for i in range(A.rows):
        degree.data[0][i] = sum(A.data[A.rowPtr[i]:A.rowPtr[i + 1]])

    T = sum(degree.data[0])

    ddt = degree.transpose().multMat(degree)
    ddt = ddt.scale(-1 / T)

    for i in range(A.rows):
        for k in range(A.rowPtr[i], A.rowPtr[i + 1]):
            j = A.colInd[k]

            ddt.data[i][j] += A.data[k]
    
    B = ddt.scale(1 / T)
    return B

def recursiveLubys(g, tau, modularity):
    A = g.adjacency

    As = [A]
    Ps = []

    while True:
        E, w = fromAdjacencyToEdge(As[-1], not modularity)

        if modularity:
            w = modularityWeights(As[-1], E)

        P = lubys(E, w)

        Ps.append(P)
        As.append(formCoarse(Ps[-1], As[-1]))
        print(As[0].rows, As[-1].rows)

        #B = modularityMatrix(As[-1])
        #if not isLaplacian(B):
        #    break

        if As[0].rows >= tau * As[-1].rows or As[-1].rows <= 2 or As[-1].rows == As[-2].rows:
            break

    return [As, Ps]

def lubys(E, w):
    if E.rows != w.dim:
        raise Exception(f"Dimension mismatch: {E.rows}x{E.columns} * {w.dim}x1")
    
    Et = E.transpose()
    
    edgeEdge = E.multMat(Et)

    labels = [-1 for _ in range(E.rows)]
    matchingCounter = 0

    for i in range(E.rows):
        edge = E.colInd[E.rowPtr[i]:E.rowPtr[i + 1]]
        weight = w.data[i]

        edgeNeighbors = _findEdgeNeighbors(i, edgeEdge)

        if _isLargestEdge(i, edgeNeighbors, w):
            labels[i] = matchingCounter
            matchingCounter += 1
    
    return _formVertexAggregate(labels, matchingCounter, Et)

def _formVertexAggregate(labels, matchingCounter, vertexEdge):
    data = [1 for _ in range(vertexEdge.rows)]
    colInd = [-1 for _ in range(vertexEdge.rows)]
    rowPtr = [0]
    nnz = 0

    aggregateCounter = matchingCounter

    for i in range(vertexEdge.rows):
        search = slice(vertexEdge.rowPtr[i], vertexEdge.rowPtr[i + 1])
        edges = vertexEdge.colInd[search]

        for e in edges:
            if labels[e] > -1:
                colInd[i] = labels[e]
                nnz += 1
                rowPtr.append(nnz)
        
        if colInd[i] == -1:
            colInd[i] = aggregateCounter
            nnz += 1
            rowPtr.append(nnz)
            aggregateCounter += 1

    return Sparse(vertexEdge.rows, aggregateCounter, data, colInd, rowPtr)

def _isLargestEdge(edge, neighbors, weights):
    for e in neighbors:
        if weights.data[e] >= weights.data[edge]:
            return False

    return weights.data[edge] >= 0

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
            # Convergence
            break
    
    meta = [len(aggregates[j]) for j in range(len(aggregates))]

    P = getVertexAggregate(X, aggregates)
    return (P, i, delta, meta)


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

        