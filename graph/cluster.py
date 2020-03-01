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

def recursiveLouvains(g):
    A = g.adjacency

    As = [A]
    Ps = []
    Q = -1

    while True:
        QOld = Q
        rand = len(As) == 1 and False
        E, w = fromAdjacencyToEdge(As[-1], rand)

        #P, Q, i = louvains(As[-1], E, 5)
        P = louvains2(As[-1], E, w)
        Q = QModularity(As[-1], P)

        Ps.append(P)
        As.append(formCoarse(Ps[-1], As[-1]))

        print(As[-1].rows, Q)

        B = modularityMatrix(As[-1])
        #if isLaplacian(B): break
        if abs(Q - QOld) == 0: 
            break
    
    return [As, Ps]

def louvains2(A, E, w):
    Et = E.transpose()
    P = _formVertexAggregate([-1 for _ in range(E.rows)], 0, Et)

    QOld = -1
    Q = QModularity(A, P)
    
    #while abs(Q - QOld) != 0:
    for k in range(2):
        QOld = Q
        
        start = time.time()
        for i in range(P.rows):
            neighbors = _findVertexNeighbors(i, A)
            if len(neighbors) == 0: continue
            '''P.visualizeShape()
            oldCluster = P.colInd[P.rowPtr[i]]
            if P.colInd.count(oldCluster) > 1:
                columnSet = set(P.colInd)
                for count, col in enumerate(columnSet):
                    if count != col: 
                        P.colInd[P.rowPtr[i]] = count
                        break
            '''
            edgeAggregate = E.multMat(P)

            clusters = [P.colInd[P.rowPtr[j]] for j in neighbors]
            dQs = [deltaQ(i, j, edgeAggregate, E, Et, w) for j in clusters]
            
            maxDq = max(dQs)
            Q += maxDq

            newCluster = clusters[dQs.index(maxDq)]
            if maxDq > 0:
                P.colInd[P.rowPtr[i]] = newCluster
        end = time.time()
        print(f"Time through vercies: {end - start}")
        
    print("Done with this iteration")
    return _reformAggregates(P)

def deltaQ(vertex, C, edgeAggregate, E, Et, w):
    aggregateEdge = edgeAggregate.transpose()

    edgesInC = []
    edgesIncidentC = []
    degreeIIncidentC = []
    degreeI = Et.colInd[Et.rowPtr[vertex]:Et.rowPtr[vertex + 1]]

    search = slice(aggregateEdge.rowPtr[C], aggregateEdge.rowPtr[C + 1])
    for entry, edge in zip(aggregateEdge.data[search], aggregateEdge.colInd[search]):
        if entry > 1: edgesInC.append(edge)
        elif entry == 1: 
            vertexPair = E.colInd[E.rowPtr[edge]:E.rowPtr[edge + 1]]
            if vertexPair[0] == vertex or vertexPair[1] == vertex:
                degreeIIncidentC.append(edge)
            edgesIncidentC.append(edge)

    edgesInC = sum([w.data[i] for i in edgesInC])
    edgesIncidentC = sum([w.data[i] for i in edgesIncidentC])
    degreeI = sum([w.data[i] for i in degreeI])
    degreeIIncidentC = sum([w.data[i] for i in degreeIIncidentC])
    m = sum(w.data)

    dQ = ((edgesInC + degreeIIncidentC) / (2 * m)) - ((edgesIncidentC + degreeI) / (2 * m)) ** 2
    dQ -= ((edgesInC / (2 * m)) - (edgesIncidentC / (2 * m)) ** 2 - (degreeI / (2 * m)) ** 2)
    return dQ

def louvains(A, E, maxIter):
    Et = E.transpose()
    P = _formVertexAggregate([-1 for _ in range(E.rows)], 0, Et)
    Q = -1

    for iterations in range(maxIter):
        QOld = Q
        start = time.time()
        for i in range(P.rows):
            Qi = QModularity(A, P)

            neighbors = _findVertexNeighbors(i, A)
            iCluster = P.colInd[P.rowPtr[i]]
            
            for j in neighbors:
                jCluster = P.colInd[P.rowPtr[j]]
                P.colInd[P.rowPtr[i]] = jCluster

                Qj = QModularity(A, P)
                
                if Qi >= Qj:
                    P.colInd[P.rowPtr[i]] = iCluster
        end = time.time()
        print(f"Time through vertcies: {end - start}")
        Q = QModularity(A, P)

        if abs(QOld - Q) == 0 and iterations > 0:
            print("Convergence")
            break
    
    P = _reformAggregates(P)
    return [P, Q, iterations]
            
def _reformAggregates(P):
    columnSet = set(P.colInd)
    columns = len(columnSet)
    columnList = list(columnSet)
    offsets = [- columnList[0]]

    for i in range(1, columns):
        offsets.append(-offsets[-1] - (columnList[i] - offsets[-1] - i))

    offsetMap = {}
    for column, offset in zip(columnList, offsets):
        offsetMap[column] = offset

    P.colInd = [i + offsetMap[i] for i in P.colInd]
    P.columns = columns
    return P

def _findVertexNeighbors(i, A):
    neighbors = A.colInd[A.rowPtr[i]:A.rowPtr[i + 1]]
    return neighbors

def recursiveLubys(g, tau, modularity):
    A = g.adjacency

    As = [A]
    Ps = []

    while True:
        E = fromAdjacencyToEdge(As[-1])

        if modularity:
            w = modularityWeights(As[-1], E)
        else:
            data = [random.random() for _ in range(E.rows)]
            w = Vector(E.rows, data)

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

        