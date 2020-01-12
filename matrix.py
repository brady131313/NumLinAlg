from abc import ABC, abstractmethod
from math import sqrt

class Vector:
    def __init__(self, dim, data = None):
        self.dim = dim

        if data and len(data) != dim:
            raise Exception("Dimension does not match data length")
        elif data and len(data) == dim:
            self.data = data
        else:
            self.data = [0.0] * dim

    def __add__(self, other):
        if self.dim != other.dim:
            raise Exception(f"Dimension mismatch: {self.dim} != {other.dim}")
        
        data = [0] * self.dim
        for i in range(self.dim):
            data[i] = self.data[i] + other.data[i]

        return Vector(self.dim, data)

    def __sub__(self, other):
        if self.dim != other.dim:
            raise Exception(f"Dimension mismatch: {self.dim} != {other.dim}")

        data = [0] * self.dim
        for i in range(self.dim):
            data[i] = self.data[i] - other.data[i]

        return Vector(self.dim, data)

    def scale(self, scalar):
        data = [0] * self.dim
        for i in range(self.dim):
            data[i] = scalar * self.data[i]

        return Vector(self.dim, data)

    def dot(self, other):
        if self.dim != other.dim:
            raise Exception(f"Dimension mismatch: {self.dim} != {other.dim}")

        sum = 0
        for i in range(self.dim):
            sum += self.data[i] * other.data[i]

        return sum

    def norm(self):
        return sqrt(self.dot(self))

    def __str__(self):
        str = f"{self.dim}x1 Vector"
        for i in range(self.dim):
            str += f"\n{self.data[i]}"

        return str



class _Matrix(ABC):
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns

    @abstractmethod
    def scale(self, scalar):
        pass

    @abstractmethod
    def transpose(self):
        pass

    @abstractmethod
    def multMat(self, other):
        pass

    @abstractmethod
    def multVec(self, other):
        pass



class Dense(_Matrix):
    def __init__(self, rows, columns, data = None):
        super().__init__(rows, columns)

        if data and (len(data) != rows or len(data[0]) != columns):
            raise Exception("Dimension does not match data shape")
        elif data and len(data) == rows and len(data[0]) == columns:
            self.data = data
        else:
            self.data = [[0.0] * self.columns for i in range(self.rows)]

    def scale(self, scalar):
        data = [[0.0] * self.columns for i in range(self.rows)]

        for i in range(self.rows):
            for j in range(self.columns):
                data[i][j] = scalar * self.data[i][j]

        return Dense(self.rows, self.columns, data)

    def transpose(self):
        data = [[0.0] * self.rows for i in range(self.columns)]

        for i in range(self.rows):
            for j in range(self.columns):
                data[j][i] = self.data[i][j]
        
        return Dense(self.columns, self.rows, data)

    def multMat(self, other):
        if self.columns != other.rows:
            raise Exception(f"Dimension mismatch: {self.rows}x{self.columns} * {other.rows}x{other.columns}")

        data = [[0.0] * other.columns for i in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.columns):
                sum = 0
                for k in range(self.columns):
                    sum += self.data[i][k] * other.data[k][j]
                data[i][j] = sum
        
        return Dense(self.rows, other.columns, data)

    def multVec(self, other):
        if self.columns != other.dim:
            raise Exception(f"Dimension mismatch: {self.rows}x{self.columns} * {other.dim}x1")

        data = [0.0] * self.rows
        for i in range(self.rows):
            for j in range(self.columns):
                data[i] += self.data[i][j] * other.data[j]

        return Vector(self.rows, data)

    def __str__(self):
        str = f"{self.rows}x{self.columns} Dense"
        for i in range(self.rows):
            str += "\n"
            for j in range(self.columns):
                str += f"{self.data[i][j]} "

        return str



class Sparse(_Matrix):
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
            self.rowPtr = []

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
                str += f"{self.getValue(i, j)} "

        return str