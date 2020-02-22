from matrix.base import Base, offset


class Dense(Base):
    def __init__(self, rows, columns, data = None):
        super().__init__(rows, columns)

        if data and (len(data) != rows or len(data[0]) != columns):
            raise Exception("Dimension does not match data shape")
        elif data and len(data) == rows and len(data[0]) == columns:
            self.data = data
        else:
            self.data = [[0.0] * self.columns for i in range(self.rows)]
    
    @classmethod
    def fromFile(cls, file, relation = False):
        entry = []
        binary = False
        checked = False

        next(file)
        info = next(file).split()
        rows, columns, = int(info[0]), int(info[1])

        data = [[0] * columns for i in range(rows)]

        for line in file:
            entry = line.split()
            if not checked:
                checked = True
                if len(entry) == 2 or relation:
                    binary = True
            
            if binary:
                data[int(entry[0]) - offset][int(entry[1]) - offset] = 1
            else:
                data[int(entry[0]) - offset][int(entry[1]) - offset] = float(entry[2])
        
        return cls(rows, columns, data)

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

    def getVector(self, column):
        if column < 0 or column >= self.columns:
            raise Exception("Column index out of range")

        data = [0] * self.rows
        for i in range(self.rows):
            data[i] = self.data[i][column]

        return Vector(self.rows, data)

    def setVector(self, column, vec):
        if column < 0 or column >= self.columns:
            raise Exception("Column index out of range")
        if vec.dim != self.rows:
            raise Exception(f"Dimension mismatch, matrix is {self.rows}x{self.columns}, vector is {vec.dim}x1")

        for i in range(self.rows):
            self.data[i][column] = vec.data[i]

    def getValue(self, row, column):
        return self.data[row][column]

    def __str__(self):
        str = f"{self.rows}x{self.columns} Dense"
        for i in range(self.rows):
            str += "\n"
            for j in range(self.columns):
                str += f"{format(self.data[i][j], '.2f')} "

        return str