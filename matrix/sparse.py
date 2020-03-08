from matrix.base import Base
from matrix.vector import Vector

class Sparse(Base):
    def __init__(self, rows, columns, data = None, colInd = None, rowPtr = None):
        super().__init__(rows, columns)

        if data and (not colInd or not rowPtr):
            raise Exception("Missing matching colInd and rowPtr for data")
        elif data and colInd and rowPtr:
            self.data = data
            self.colInd = colInd
            self.rowPtr = rowPtr
        elif not data and colInd and rowPtr:
            self.data = [1] * len(colInd)
            self.colInd = colInd
            self.rowPtr = rowPtr
        else:
            self.data = []
            self.colInd = []
            self.rowPtr = [0]

    @classmethod
    def fromFile(cls, file, offset = 0, relation = False, graph = False):
        data = []
        colInd = []
        rowPtr = [0]
        nnz = 0
        binary = False

        next(file)

        info = next(file).split()
        rows, columns = int(info[0]), int(info[1])

        entries = [[] for i in range(rows)]

        for line in file:
            temp = line.split()
            row, column = int(temp[0]), int(temp[1])
            row -= offset
            column -= offset

            num = float(temp[2]) if len(temp) == 3 else None

            entry = (row, column, num) if num else (row, column)

            if not graph:
                entries[row].append(entry)
            elif graph and row != column:
                entries[row].append(entry)

        if len(entries[0]) == 2 or relation:
            binary = True

        for i in range(rows):
            for j in range(len(entries[i])):
                if binary:
                    data.append(1)
                else:
                    data.append(entries[i][j][2])

                colInd.append(entries[i][j][1])
                nnz += 1

            rowPtr.append(nnz)
        
        return cls(rows, columns, data, colInd, rowPtr)
    
    @classmethod
    def fromDense(cls, dense):
        data = []
        colInd = []
        rowPtr = [0]
        nnz = 0

        for i in range(dense.rows):
            for j in range(dense.columns):
                if dense.data[i][j] != 0:
                    data.append(dense.data[i][j])
                    colInd.append(j)
                    nnz += 1
            rowPtr.append(nnz)

        return cls(dense.rows, dense.columns, data, colInd, rowPtr)


    def scale(self, scalar):
        data = [0.0] * len(self.data)

        for i in range(len(self.data)):
            data[i] = scalar * self.data[i]

        return Sparse(self.rows, self.columns, data, self.colInd, self.rowPtr)

    def transpose(self):
        nnz = self.rowPtr[self.rows]
        colPtr = [0] * (self.columns + 1)
        rowInd = [0] * nnz
        data = [0] * nnz

        for i in range(nnz):
            colPtr[self.colInd[i]] += 1

        sum = 0
        for i in range(self.columns):
            temp = colPtr[i]
            colPtr[i] = sum
            sum += temp
        colPtr[self.columns] = nnz

        for i in range(self.rows):
            for j in range(self.rowPtr[i], self.rowPtr[i + 1]):
                col = self.colInd[j]
                dest = colPtr[col]

                rowInd[dest] = i
                data[dest] = self.data[j]

                colPtr[col] += 1
        
        last = 0
        for i in range(self.columns + 1):
            temp = colPtr[i]
            colPtr[i] = last
            last = temp

        return Sparse(self.columns, self.rows, data, rowInd, colPtr)

    def multMat(self, other):
        n = self.rows
        m = other.columns
        rowPtr = self.genShape(other)
        colInd = [0] * rowPtr[n]
        data = [0.0] * rowPtr[n]
        nnz = 0
        localToGlobal = [0] * m
        globalToLocal = [-1] * m

        for i in range(self.rows):
            localCounter = 0
            head = -2

            for p in range(self.rowPtr[i], self.rowPtr[i + 1]):
                k = self.colInd[p]
                v = self.data[p]

                for q in range(other.rowPtr[k], other.rowPtr[k + 1]):
                    j = other.colInd[q]

                    localToGlobal[j] += v * other.data[q]
                    
                    if (globalToLocal[j] < 0):
                        globalToLocal[j] = head
                        head = j
                        localCounter += 1
                                        
            for counter in range(localCounter):
                if localToGlobal[head] != 0:
                    colInd[nnz] = head
                    data[nnz] = localToGlobal[head]
                    nnz += 1

                temp = head
                head = globalToLocal[head]
                globalToLocal[temp] = -1
                localToGlobal[temp] = 0
            
            rowPtr[i + 1] = nnz 

        return Sparse(n, m, data, colInd, rowPtr)

    def genShape(self, other):
        n = self.rows
        m = other.columns
        localToGlobal = [0] * m
        globalToLocal = [-1] * m
        rowPtr = [0] * (n + 1)
        
        for i in range(self.rows):
            localCounter = 0

            for p in range(self.rowPtr[i], self.rowPtr[i + 1]):
                k = self.colInd[p]

                for q in range(other.rowPtr[k], other.rowPtr[k + 1]):
                    j = other.colInd[q]

                    if globalToLocal[j] < 0:
                        globalToLocal[j] = localCounter
                        localToGlobal[localCounter] = j
                        localCounter += 1
            
            rowPtr[i + 1] = rowPtr[i] + localCounter
            for counter in range(localCounter):
                globalToLocal[localToGlobal[counter]] = -1
        
        return rowPtr

    def multVec(self, other):
        if self.columns != other.dim:
            raise Exception(f"Dimension mismatch: {self.rows}x{self.columns} * {other.dim}x1")

        data = [0.0] * self.rows
        for i in range(self.rows):
            for k in range(self.rowPtr[i], self.rowPtr[i + 1]):
                j = self.colInd[k]
                #print(self.data[self.rowPtr[i]:self.rowPtr[i + 1]])
                data[i] += self.data[k] * other.data[j]

        return Vector(self.rows, data)

    def getValue(self, row, column):
        if len(self.data) == 0:
            return 0

        search = self.colInd[self.rowPtr[row]:self.rowPtr[row + 1]]
        try:
            k = search.index(column)
        except ValueError:
            return 0

        if self.data:
            return self.data[self.rowPtr[row] + k]
        else:
            return 1

    def __str__(self):
        str = f"{self.rows}x{self.columns} Sparse"
        for i in range(self.rows):
            str += "\n"
            for j in range(self.columns):
                str += f"{format(self.getValue(i, j), '.2f')} "

        return str