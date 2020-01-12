import matrix

data = [1, 3, 2]
colInd = [0, 1, 0]
rowPtr = [0, 1, 2, 2, 3]

data2 = [1, 1, 3]
colInd2 = [0, 1, 2]
rowPtr2 = [0, 2, 3]

s1 = matrix.Sparse(4, 2, data, colInd, rowPtr)
#s2 = matrix.Sparse(2, 3, data2, colInd2, rowPtr2)

print(s1)
print(s1.transpose())
#print(s2)
#print(s1.multMat(s2))
