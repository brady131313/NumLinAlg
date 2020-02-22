from operator import itemgetter
import numpy as np
import random

from matrix import Sparse, Vector, offset
from linalg import l1Smoother

class Graph:
    def __init__(self, adjacency = None, edgeVertex = None):
        self.adjacency = adjacency
        self.edgeVertex = edgeVertex

    @classmethod
    def fromFile(cls, file):
        adjacency = Sparse.fromFile(file, True, True)

        #Go back to beginning of file
        file.seek(0)
        edgeVertex = fromFileToEdge(file)

        return cls(adjacency, edgeVertex)

    def getVertexEdge(self):
        return self.edgeVertex.transpose()

    def getEdgeEdge(self):
        return self.edgeVertex.multMat(self.getVertexEdge())

    def getDegree(self):
        return l1Smoother(self.adjacency)

    def getLaplacian(self):
        L = self.getVertexEdge().multMat(self.edgeVertex)

        for i in range(L.rows):
            for k in range(L.rowPtr[i], L.rowPtr[i + 1]):
                j = L.colInd[k]
                
                if i != j:
                    L.data[k] = - L.data[k]

        return L
    
    def getRandomWeights(self):
        data = [random.random() for _ in range(self.edgeVertex.rows)]
        return Vector(self.edgeVertex.rows, data)


def getVertexAggregate2(vertcies, clusters):
    data = []
    colInd = []
    rowPtr = [0]
    nnz = 0

    for i in range(vertcies):
        for j in range(len(clusters)):
            if i in clusters[j]:
                data.append(1)
                colInd.append(j)
                nnz += 1
                rowPtr.append(nnz)
    
    return Sparse(nnz, len(clusters), data, colInd, rowPtr)

def getVertexAggregate(X, clusters):
    data = []
    colInd = []
    rowPtr = [0]
    nnz = 0

    X = [Vector(len(x), list(x)) for x in X]
    for i in range(len(clusters)):
        clusters[i] = [Vector(len(c), list(c)) for c in clusters[i]]

    for x in X:
        aggregate = _findAggregate(x, clusters)

        data.append(1)
        colInd.append(aggregate)
        nnz += 1
        rowPtr.append(nnz)

    return Sparse(nnz, len(clusters), data, colInd, rowPtr)

def _findAggregate(x, clusters):
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            if x.equal(clusters[i][j]):
                return i
            

def formCoarse(P, A):
    return P.transpose().multMat(A).multMat(P)


def formVertexToK1Aggregate(Ps, k):
    if k > len(Ps):
        raise Exception("Not enough relations")

    pi = Ps[0]

    for i in range(1, k):
        pi = pi.multMat(Ps[i])

    return pi


def fromAdjacencyToEdge(A):
    duplicates = set()
    entries = []

    for i in range(A.rows):
        for k in range(A.rowPtr[i], A.rowPtr[i + 1]):
            j = A.colInd[k]
            entry = (i, j)

            if entry not in duplicates and entry[::-1] not in duplicates and entry[0] != [1]:
                duplicates.add(entry)
                entries.append(sorted(entry))

    entries = sorted(entries, key=itemgetter(0))

    return _processEntries(len(entries), A.columns, entries)

def fromFileToEdge(file):
    data = []
    colInd = []
    rowPtr = []
    nnz = 0
    duplicates = set()
    entries = []

    next(file)

    info = next(file).split()
    rows, columns = int(info[0]), int(info[1])

    for line in file:
        temp = line.split()
        #Lists in python are not hashable so store in tuple
        entry = (int(temp[0]) - offset, int(temp[1]) - offset)
        #Check if tuple has already been read
        if entry not in duplicates and entry[::-1] not in duplicates and entry[0] != entry[1]:
            duplicates.add(entry)
            entries.append(sorted(entry))

    #Sort entries by row so they can be added to CSR 
    entries = sorted(entries, key=itemgetter(0))
    
    return _processEntries(len(entries), columns, entries)

def _processEntries(rows, columns, entries):
    data = []
    colInd = []
    rowPtr = []
    nnz = 0

    for i, entry in enumerate(entries):
        rowPtr.append(nnz)

        data.append(1)
        colInd.append(entry[0])

        data.append(1)
        colInd.append(entry[1])

        nnz += 2
    rowPtr.append(nnz)

    return Sparse(rows, columns, data, colInd, rowPtr)

        

