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

        


def fromFileToEdge(file):
    data = []
    colInd = []
    rowPtr = [0]
    nnz = 0
    duplicates = set()
    entries = []

    next(file)

    info = next(file).split()
    rows, columns = int(info[0]) - matrix.offset, int(info[1]) - matrix.offset

    for line in file:
        temp = line.split()
        #Lists in python are not hashable so store in tuple
        entry = (int(temp[0]), int(temp[1]))
        #Check if tuple has already been read
        if entry not in duplicates and entry[::-1] not in duplicates:
            duplicates.add(entry)
            entries.append(sorted(entry))

    #Sort entries by row so they can be added to CSR 
    entries = sorted(entries, key=itemgetter(0))
    
    last = 0
    for i, entry in enumerate(entries):
        data.append(1)
        colInd.append(entry[0])

        data.append(1)
        colInd.append(entry[1])

        if last != i:
            rowPtr.append(nnz)

        nnz += 2
        last = i
    
    rowPtr.append(nnz)

    return matrix.Sparse(rows, columns, data, colInd, rowPtr)

        

