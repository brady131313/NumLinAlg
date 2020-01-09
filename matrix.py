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
    def __init__(self, rows, columns):
        super().__init__(rows, columns)

    def print(self):
        print(self.rows, "x", self.columns, " sparse", sep='')