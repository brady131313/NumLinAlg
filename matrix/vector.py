from math import sqrt
import random

class Vector:
    def __init__(self, dim, data = None):
        self.dim = dim

        if data and len(data) != dim:
            raise Exception("Dimension does not match data length")
        elif data and len(data) == dim:
            self.data = data
        else:
            self.data = [0.0] * dim

    @classmethod
    def fromRandom(cls, dim, min, max):
        if dim <= 0:
            raise Exception("Dimension must be greater than 0")

        data = [0] * dim
        for i in range(dim):
            data[i] = random.randint(min, max)
        
        return cls(dim, data)

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

    def equal(self, other):
        if self.dim != other.dim:
            raise Exception(f"Dimension mismatch: {self.dim} != {other.dim}")

        for i in range(self.dim):
            if self.data[i] != other.data[i]:
                return False

        return True

    def scale(self, scalar):
        data = [0] * self.dim
        for i in range(self.dim):
            data[i] = scalar * self.data[i]

        return Vector(self.dim, data)

    def elementWiseMult(self, other):
        if self.dim != other.dim:
            raise Exception(f"Dimension mismatch: {self.dim} != {other.dim}")

        data = [0] * self.dim
        for i in range(self.dim):
            data[i] = self.data[i] * other.data[i]

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