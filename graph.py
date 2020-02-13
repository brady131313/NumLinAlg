from operator import itemgetter
import matrix

class Graph(matrix.Sparse):
    def __init__(self, rows, columns, data = None, colInd = None, rowPtr = None):
        super().__init__(rows, columns, data, colInd, rowPtr)

    @classmethod
    def fromFileToAdjacency(cls, file):
        return super().fromFile(file, True)

    @classmethod
    def fromFileToEdge(cls, file):
        data = []
        colInd = []
        rowPtr = [0]
        nnz = 0
        duplicates = set()
        entries = []

        next(file)

        info = next(file).split()
        rows, columns = int(info[0]), int(info[1])

        for line in file:
            temp = line.split()
            entry = (int(temp[0]), int(temp[1]))
            if entry not in duplicates and entry[::-1] not in duplicates:
                duplicates.add(entry)
                entries.append(sorted(entry))

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

        return cls(rows, columns, data, colInd, rowPtr)

        

