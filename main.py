import matrix

data = [3, 1, 2, 4]
colInd = [0, 0, 2, 1]
rowPtr = [0, 1, 3, 4]

data2 = [2, 1, 3, 1]
colInd2 = [1, 1, 0, 2]
rowPtr2 = [0, 1, 2, 4]

s1 = matrix.Sparse(3, 3, data, colInd, rowPtr)
s2 = matrix.Sparse(3, 3, data2, colInd2, rowPtr2)

print(s1)
print(s2)
print(s1.multMat(s2))
