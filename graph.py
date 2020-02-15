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
    rowPtr = []
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
        if entry not in duplicates and entry[::-1] not in duplicates and entry[0] != entry[1]:
            duplicates.add(entry)
            entries.append(sorted(entry))

    #Sort entries by row so they can be added to CSR 
    entries = sorted(entries, key=itemgetter(0))
    print(len(entries))
    print(len(duplicates))
    print(entries)
    
    for i, entry in enumerate(entries):
        rowPtr.append(nnz)

        data.append(1)
        colInd.append(entry[0])

        data.append(1)
        colInd.append(entry[1])

        nnz += 2
    
    rowPtr.append(nnz)    

    return matrix.Sparse(len(entries), columns, data, colInd, rowPtr)

        

