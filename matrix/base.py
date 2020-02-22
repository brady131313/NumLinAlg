from abc import ABC, abstractmethod

offset = 1

class Base(ABC):
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

    @abstractmethod
    def getValue(self, row, column):
        pass

    def visualizeShape(self):
        str = ""
        for i in range(self.rows):
            str += "\n"
            for j in range(self.columns):
                value = self.getValue(i, j)
                str += "■" if value != 0 else "▫"
                str += " "
        
        print(str)