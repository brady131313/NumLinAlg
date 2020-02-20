from operator import itemgetter
import matrix
import decomp

class Graph:
    def __init__(self, adjacency = None, edgeVertex = None):
        self.adjacency = adjacency
        self.edgeVertex = edgeVertex

    @classmethod
    def fromFile(cls, file):
        adjacency = matrix.Sparse.fromFile(file, True, True)

        #Go back to beginning of file
        file.seek(0)
        edgeVertex = fromFileToEdge(file)

        return cls(adjacency, edgeVertex)

    def getVertexEdge(self):
        return self.edgeVertex.transpose()

    def getEdgeEdge(self):
        return self.edgeVertex.multMat(self.getVertexEdge())

    def getDegree(self):
        return decomp.l1Smoother(self.adjacency)

    def getLaplacian(self):
        L = self.getVertexEdge().multMat(self.edgeVertex)

        for i in range(L.rows):
            for k in range(L.rowPtr[i], L.rowPtr[i + 1]):
                j = L.colInd[k]
                
                if i != j:
                    L.data[k] = - L.data[k]

        return L



def getVertexAggregate(coords, clusters):
    data = []
    colInd = []
    rowPtr = [0]
    nnz = 0

    for i in range(len(coords)):
        for j in range(len(clusters)):
            for k in range(len(clusters[j])):
                if coords[i].equal(clusters[j][k]):
                    data.append(1)
                    colInd.append(j)
        nnz += 1
        rowPtr.append(nnz)


    return matrix.Sparse(len(coords), len(clusters), data, colInd, rowPtr)

def formCoarse(P, A):
    return P.transpose().multMat(A).multMat(P)


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
        entry = (int(temp[0]) - matrix.offset, int(temp[1]) - matrix.offset)
        #Check if tuple has already been read
        if entry not in duplicates and entry[::-1] not in duplicates and entry[0] != entry[1]:
            duplicates.add(entry)
            entries.append(sorted(entry))

    #Sort entries by row so they can be added to CSR 
    entries = sorted(entries, key=itemgetter(0))
    
    for i, entry in enumerate(entries):
        rowPtr.append(nnz)

        data.append(1)
        colInd.append(entry[0])

        data.append(1)
        colInd.append(entry[1])

        nnz += 2
    
    rowPtr.append(nnz)    

    return matrix.Sparse(len(entries), columns, data, colInd, rowPtr)

        

